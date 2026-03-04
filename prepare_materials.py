import time
from pathlib import Path
import glob

from pcmfile import *
from quantize import *
from window import *
from mdct import *
from pacfile import *
from rmcfile import RMCFile

def quantize_td_fp(inFilename, outFilename):
    inFile = PCMFile(inFilename)
    outFile = PCMFile(outFilename)
    codingParams = inFile.OpenForReading()
    codingParams.nSamplesPerBlock = 1024
    outFile.OpenForWriting(codingParams)

    while True:
        data = inFile.ReadDataBlock(codingParams)
        if not data: break  
        for iCh in range(codingParams.nChannels):
          for i in range(0, len(data[iCh])):
            sf = ScaleFactor(data[iCh][i])
            mant = MantissaFP(data[iCh][i], sf)
            data[iCh][i] = DequantizeFP(sf, mant)
        outFile.WriteDataBlock(data,codingParams)

    inFile.Close(codingParams)
    outFile.Close(codingParams)

      
def quantize_td_bfp(inFilename, outFilename):
    inFile = PCMFile(inFilename)
    outFile = PCMFile(outFilename)
    codingParams= inFile.OpenForReading()
    codingParams.nSamplesPerBlock = 1024
    outFile.OpenForWriting(codingParams)

    while True:
        data=inFile.ReadDataBlock(codingParams)
        if not data: break  # we hit the end of the input file
        batch_size = 1
        for iCh in range(codingParams.nChannels):
          l = len(data[iCh])
          for i in range(0, l, batch_size):
            end = min(i+batch_size, l)
            batch = data[iCh][i:end]
            max_val = batch[np.argmax(np.abs(batch))]
            sf = ScaleFactor(max_val)
            mant = vMantissa(batch, sf)
            data[iCh][i:end] = vDequantize(sf, mant)
        outFile.WriteDataBlock(data,codingParams)

    inFile.Close(codingParams)
    outFile.Close(codingParams)

def quantize_fd_bfp(inFilename, outFilename):
    inFile = PCMFile(inFilename)
    outFile = PCMFile(outFilename)
    codingParams= inFile.OpenForReading()
    codingParams.nSamplesPerBlock = 2048

    outFile.OpenForWriting(codingParams)

    prior_block = np.zeros((codingParams.nChannels, codingParams.nSamplesPerBlock))
    overlapAndAdd = np.zeros((codingParams.nChannels, codingParams.nSamplesPerBlock))
    
    first = True

    while True:
        data=inFile.ReadDataBlock(codingParams)
        if not data: 
            # we hit the end of the input file 
            # (3a) Create a buffer
            buf = np.zeros((codingParams.nChannels, codingParams.nSamplesPerBlock))
            for iCh in range(codingParams.nChannels):
                # (3a) Make a block using prior block w/ padded zeros at the end
                block = np.concat([prior_block[iCh], np.zeros(codingParams.nSamplesPerBlock)])
                # (3a) Window block
                block = KBDWindow(block)
                # (3b) Perform MDCT
                block = MDCT(block, codingParams.nSamplesPerBlock, codingParams.nSamplesPerBlock)
                for i in range(len(block)):
                    # (3d) Perform quantization
                    scale = ScaleFactor(block[i])
                    mant = Mantissa(block[i], scale)
                    # (3d) Perform dequantization
                    block[i] = Dequantize(scale, mant)
                # (3b) Perform IMDCT
                block = IMDCT(block, codingParams.nSamplesPerBlock, codingParams.nSamplesPerBlock)
                # (3a) Window block again
                block = KBDWindow(block)
                # (3a) Add left side to overlapAndAdd and write to output
                output = overlapAndAdd[iCh] + block[:codingParams.nSamplesPerBlock]
                # (3a) Set data to output
                buf[iCh] = output
            # (3a) Flush and break
            outFile.WriteDataBlock(buf,codingParams)
            break
            
        for iCh in range(codingParams.nChannels):
            # (3a) Concat block to prior block
            block = np.concat([prior_block[iCh], data[iCh]])
            # (3a) Reset prior block
            prior_block[iCh] = data[iCh]
            # (3a)Window block
            block = KBDWindow(block)
            # (3b) Perform MDCT
            block = MDCT(block, codingParams.nSamplesPerBlock, codingParams.nSamplesPerBlock)
            for i in range(len(block)):
                # (3d) Perform quantization
                scale = ScaleFactor(block[i])
                mant = Mantissa(block[i], scale)
                # (3d) Perform dequantization
                block[i] = Dequantize(scale, mant)
            # (3b) Perform IMDCT
            block = IMDCT(block, codingParams.nSamplesPerBlock, codingParams.nSamplesPerBlock)
            # (3a) Window block again
            block = KBDWindow(block)
            # (3a) Add left side to overlapAndAdd and write to output
            output = overlapAndAdd[iCh] + block[:codingParams.nSamplesPerBlock]
            # (3a) Set data to output
            data[iCh] = output
            # (3a) Set overlapAndAdd to right side
            overlapAndAdd[iCh] = block[codingParams.nSamplesPerBlock:]

        # (3a) Remove delay on first block from zero padding
        if not first:
            outFile.WriteDataBlock(data,codingParams)
        else:
            first = False
    # end loop over reading/writing the blocks

    # close the files
    inFile.Close(codingParams)
    outFile.Close(codingParams)

def pac(inFilename, outFilename, rate_kb=128):
    codedFilename = f"coded/{Path(outFilename).stem}.pac"
    # print ("\nRunning the PAC coder (" + inFilename +
    #     " -> " + codedFilename + " -> " +
    #     outFilename+"):")
    # elapsed = time.time()

    for iDir, Direction in enumerate(("Encode", "Decode")):

        # create the audio file objects
        if Direction == "Encode":
            print( "\n\tEncoding input PCM file...", end="")
            inFile= PCMFile(inFilename)
            outFile = PACFile(codedFilename)
        else: # "Decode"
            print( "\n\tDecoding coded PAC file...",end="")
            inFile = PACFile(codedFilename)
            outFile= PCMFile(outFilename)
        # only difference is file names and type of AudioFile object

        # open input file
        codingParams=inFile.OpenForReading()  # (includes reading header)
        

        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)
            codingParams.nMDCTLines = 1024
            codingParams.nScaleBits = 3
            codingParams.nMantSizeBits = 5
            codingParams.targetBitsPerSample = rate_kb * 1000 / codingParams.sampleRate
            print(f"Target bits/sample: {codingParams.targetBitsPerSample:0.1f}")
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
        else: # "Decode"
            # set PCM parameters (the rest is same as set by PAC file on open)
            codingParams.bitsPerSample = 16
        # only difference is in setting up the output file parameters


        # open the output file
        outFile.OpenForWriting(codingParams) # (includes writing header)

        # Read the input file and pass its data to the output file to be written
        while True:
            data = inFile.ReadDataBlock(codingParams)
            if not data: break  # we hit the end of the input file
            outFile.WriteDataBlock(data,codingParams)
            # Progress update
            print( ".",end="")  # just to signal how far we've gotten to user
        # end loop over reading/writing the blocks

        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)
        # end of loop over Encode/Decode

        # elapsed = time.time()-elapsed
        # print( "\nDone with Encode/Decode test\n")
        # print( elapsed ," seconds elapsed")

def rmc(inFilename, outFilename, rate_kb=128):
    codedFilename = f"coded/{Path(inFilename).stem}_rmc_{rate_kb}kbps.rmc"

    for Direction in ("Encode", "Decode"):
        if Direction == "Encode":
            print("\n\tEncoding input PCM file...", end="")
            inFile = PCMFile(inFilename)
            outFile = RMCFile(codedFilename)
        else:
            print("\n\tDecoding coded RMC file...", end="")
            inFile = RMCFile(codedFilename)
            outFile = PCMFile(outFilename)

        codingParams = inFile.OpenForReading()

        if Direction == "Encode":
            codingParams.nMDCTLines = 1024
            codingParams.nScaleBits = 3
            codingParams.nMantSizeBits = 5
            codingParams.targetBitsPerSample = rate_kb * 1000 / codingParams.sampleRate
            print(f"Target bits/sample: {codingParams.targetBitsPerSample:0.1f}")
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
        else:
            codingParams.bitsPerSample = 16

        outFile.OpenForWriting(codingParams)

        while True:
            data = inFile.ReadDataBlock(codingParams)
            if not data: break
            outFile.WriteDataBlock(data, codingParams)
            print(".", end="")

        inFile.Close(codingParams)
        outFile.Close(codingParams)


if __name__ == "__main__":
    reference_files = glob.glob("reference/*.wav")
    for i, file in enumerate(reference_files):
        reference_name = Path(file).stem
        print(f"[{i + 1}/{len(reference_files)}] Running for {file}")
        print("Time domain FP")
        quantize_td_fp(file, f"coded/{reference_name}_s3m5.wav")
        print("Time domain BFP")
        quantize_td_bfp(file, f"coded/{reference_name}_b1s3m5.wav")
        print("Freq domain BFP")
        quantize_fd_bfp(file, f"coded/{reference_name}_fb1s3m5.wav")
        print("PAC 128 kb/s/ch")
        pac(file, f"coded/{reference_name}_128kbps.wav", rate_kb=128)
        print("PAC 192 kb/s/ch")
        pac(file, f"coded/{reference_name}_192kpbs.wav", rate_kb=192)
        print("Done\n")
        
        
        
        