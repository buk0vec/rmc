import numpy as np

PRED_MAP = {
    None: 0,
    "quarter": 1,
    "half": 2,
    "bar": 3
}

INV_PRED_MAP = {
    0: None,
    1: "quarter",
    2: "half",
    3: "bar"
}

def get_best_region(input_block, coding_params, buffer, threshold=0.8):
    """
    Find the best rhythmic prediction region for the input block.
    
    Parameters:
    -----------
    input_block : np.ndarray
        2048 samples to encode
    coding_params : object
        Contains tempo, sample_rate, etc.
    buffer : np.ndarray
        Previous audio history (1 bar + padding)
    padding : int
        Search range padding in samples (default: 200)
    threshold : float
        MSE threshold ratio for rejecting prediction (default: 0.8)
    
    Returns:
    --------
    tuple: (range_type, data_to_encode, relative_offset)
        range_type: str - "quarter", "half", "bar", or None
        data_to_encode: np.ndarray - residual if prediction used, otherwise input_block
        relative_offset: int - offset from center (0 if no prediction used)
    """
    
    # Calculate baseline energy
    original_energy = np.var(input_block)
    
    block_size = len(input_block)
    
    # Initialize results storage
    results = {}
    
    #search range 
    padding = coding_params.search_range

    # Search each range type
    range_configs = {
        'quarter': coding_params.numSamplesQuarterNote ,
        'half': coding_params.numSamplesHalfBar,
        'bar': coding_params.numSamplesBar
    }


    for range_type, qn_multiplier in range_configs.items():
        
        # Determine center offset
        center_offset = qn_multiplier
        
        # Define search window using negative indices from end of buffer
        # -center_offset is the nominal prediction point (e.g. exactly 1 quarter note ago)
        search_start = -center_offset - padding
        search_end = -center_offset + padding

        # Ensure we don't go out of buffer bounds
        if search_start < -len(buffer):
            continue  # Skip this range if not enough history in buffer

        # Sliding correlation search
        best_correlation = -np.inf
        best_sample_offset = None

        for sample_offset in range(search_start, search_end + 1):
            # Extract candidate region from buffer
            candidate_region = buffer[sample_offset : sample_offset + block_size]

            # Calculate normalized correlation
            correlation = np.corrcoef(input_block, candidate_region)[0, 1]
            # Track best match
            if correlation > best_correlation:
                best_correlation = correlation
                best_sample_offset = sample_offset

        if best_sample_offset is None:
            best_sample_offset = -center_offset

        # Calculate residual and MSE
        predicted_block = buffer[best_sample_offset : best_sample_offset + block_size]
        residual = input_block - predicted_block
        mse = np.mean(residual ** 2)

        # Calculate relative offset from nominal center for block header encoding.
        # Decoder reconstructs as buffer[-start_offset + relative_offset], so
        # relative_offset = best_sample_offset + center_offset gives the same index.
        relative_offset = best_sample_offset + center_offset
        
        # Store results
        results[range_type] = {
            'sample_offset': best_sample_offset,
            'relative_offset': relative_offset,
            'residual': residual,
            'mse': mse,
            'correlation': best_correlation
        }
    
    # Check if we found any valid results
    if not results:
        # No valid search performed, encode input directly
        return (None, input_block, 0)
    
    # Find best range across all types
    best_range = min(results.keys(), key=lambda x: results[x]['mse'])
    best_mse = results[best_range]['mse']
    best_residual = results[best_range]['residual']
    best_relative_offset = results[best_range]['relative_offset']
    
    # Evaluate against threshold
    if best_mse < threshold * original_energy:
        # Prediction is helpful - encode residual
        return (best_range, best_residual, best_relative_offset)
    else:
        # Prediction doesn't help - encode original input
        return (None, input_block, 0)