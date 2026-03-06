import numpy as np

def get_best_region(input_block, coding_params, buffer, padding=200, threshold=0.8):
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
    
    # Calculate temporal parameters
    samples_per_qn = int((60.0 / coding_params.tempo) * coding_params.sample_rate)
    block_size = len(input_block)
    
    # Initialize results storage
    results = {}
    
    # Search each range type
    range_configs = {
        'quarter': 1,
        'half': 2,
        'bar': 4
    }
    
    for range_type, qn_multiplier in range_configs.items():
        
        # Determine center offset
        center_offset = qn_multiplier * samples_per_qn
        
        # Define search window
        search_start = center_offset - padding
        search_end = center_offset + padding
        
        # Ensure we don't go out of buffer bounds
        if search_end + block_size > len(buffer):
            continue  # Skip this range if not enough buffer
        
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
        
        # Calculate residual and MSE
        predicted_block = buffer[best_sample_offset : best_sample_offset + block_size]
        residual = input_block - predicted_block
        mse = np.mean(residual ** 2)
        
        # Calculate relative offset (for block header encoding)
        relative_offset = best_sample_offset - center_offset
        
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