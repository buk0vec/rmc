import numpy as np

def get_search_offsets(time_signature):
    """
    Generate meter related search offsets based on meter.
    
    Parameters:
    -----------
    time_signature : tuple
        (beats_per_bar, beat_unit), e.g., (4, 4), (3, 4), (12, 8), (7, 4)
    
    Returns:
    --------
    list of int: [num_beat_units, ...]
        Number of beat units to look backward for each search offset

    Examples:
    ---------
    (4, 4) → [1, 2, 4]
    (3, 4) → [1, 2, 3]
    (12, 8) → [3, 6, 12]
    (7, 4) → [1, 2, 3, 7]
    """
    beats_per_bar, beat_unit = time_signature
    offsets = []
    
    # COMPOUND METERS 
    if beat_unit == 8 and beats_per_bar % 3 == 0:
        compound_beats = beats_per_bar // 3
        
        # 1 beat (3 eighth notes)
        offsets.append(3)
        
        # 2 beats (6 eighth notes)
        if compound_beats > 2:
            offsets.append(6)
        
        # Half bar
        if compound_beats % 2 == 0 and compound_beats > 2:
            offsets.append(beats_per_bar // 2)
        
        # Full bar
        offsets.append(beats_per_bar)
    
    # SIMPLE METERS 
    else:
        # 1 beat
        offsets.append(1)
        
        # 2 beats
        if beats_per_bar >= 2:
            offsets.append(2)
        # Half bar
        if beats_per_bar % 2 == 0:
            offsets.append(beats_per_bar // 2)
        elif beats_per_bar > 3:
            offsets.append(beats_per_bar // 2)
        
        # Full bar
        offsets.append(beats_per_bar)
    

    offsets = sorted(list(set(offsets)))
    
    return offsets



def get_best_region(input_block, coding_params, buffer, padding=200, threshold=0.8):
    
    original_energy = np.var(input_block)
    
    # Get time signature from header
    time_signature = (coding_params.beats_per_bar, coding_params.beat_unit)
    
    # Calculate samples per beat unit
    samples_per_quarter_note = int((60.0 / coding_params.tempo) * coding_params.sample_rate)
    samples_per_beat_unit = int(samples_per_quarter_note * (4.0 / time_signature[1]))
    
    block_size = len(input_block)
    results = {}
    
    # Get adaptive search offsets (now just numbers)
    search_offsets = get_search_offsets(time_signature)
    
    for beat_units in search_offsets:
        
        center_offset = beat_units * samples_per_beat_unit
        search_start = center_offset - padding
        search_end = center_offset + padding
        
        if search_end + block_size > len(buffer):
            continue
        
        # Sliding correlation search
        best_correlation = -np.inf
        best_sample_offset = None
        
        for sample_offset in range(search_start, search_end + 1):
            candidate_region = buffer[sample_offset : sample_offset + block_size]
            correlation = np.corrcoef(input_block, candidate_region)[0, 1]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_sample_offset = sample_offset
        
        # Calculate residual and MSE
        predicted_block = buffer[best_sample_offset : best_sample_offset + block_size]
        residual = input_block - predicted_block
        mse = np.mean(residual ** 2)
        relative_offset = best_sample_offset - center_offset
        
        # Use beat_units as key
        results[beat_units] = {
            'sample_offset': best_sample_offset,
            'relative_offset': relative_offset,
            'residual': residual,
            'mse': mse,
            'correlation': best_correlation
        }
    
    if not results:
        return (None, input_block, 0)
    
    # Find best by minimum MSE
    best_beat_units = min(results.keys(), key=lambda x: results[x]['mse'])
    best_mse = results[best_beat_units]['mse']
    best_residual = results[best_beat_units]['residual']
    best_relative_offset = results[best_beat_units]['relative_offset']
    
    if best_mse < threshold * original_energy:
        return (best_beat_units, best_residual, best_relative_offset)
    else:
        return (None, input_block, 0)