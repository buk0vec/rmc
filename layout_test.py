import sys, bisect
sys.path.insert(0, '.')
from simple_run import detectTransients
import soundfile as sf

halfN=1024; halfN_short=128
LONG,START,SHORT,STOP,MEDIUM=0,1,2,3,4
BN={0:'LONG',1:'START',2:'SHORT',3:'STOP',4:'MEDIUM'}

events = detectTransients('Van_124.wav', verbose=False, return_events=True, forceBlockSize=halfN)
positions = []
last = -2176
for e in events:
    si = int(e['sample_index'])
    if si - last >= 2176:
        positions.append(si)
        last = si

info = sf.info('Van_124.wav')
total = info.frames + halfN
pos=0; prev=LONG; queue=[]; log=[]

while pos < total:
    tidx = bisect.bisect_left(positions, pos)
    if tidx < len(positions) and positions[tidx] < pos + 2*halfN:
        raw = positions[tidx] - pos
        k = max(0, min(7, (raw - halfN) // halfN_short))
        bs = (7+k)*halfN_short
    else:
        k=-1; bs=halfN_short

    if queue:
        item=queue.pop(0); bt,cb,ca=item['type'],item['b'],item['a']
    elif prev in (LONG,STOP):
        if k>=0:
            bt=START; cb=bs; ca=halfN
            queue.append({'type':MEDIUM,'a':(7+k)*128,'b':128})
            queue.append({'type':SHORT,'a':halfN_short,'b':halfN_short})
            queue.append({'type':STOP,'a':halfN,'b':halfN})
        else:
            bt=LONG; cb=halfN; ca=halfN
    elif prev in (START,MEDIUM):
        bt=SHORT; cb=128; ca=128
    elif prev==SHORT:
        bt=SHORT; cb=128; ca=128
    else:
        bt=LONG; cb=halfN; ca=halfN

    log.append((bt,pos,cb,ca,k if bt==START else -1))
    prev=bt
    pos+=cb

lo_s=int(6.0*44100); hi_s=int(7.3*44100)
print("type      start      end    len  win_a  win_b    sec    k")
print("-"*62)
for bt,bpos,blen,bca,bk in log:
    if bpos+blen<lo_s: continue
    if bpos>hi_s: break
    ks=str(bk) if bk>=0 else ''
    print("%-7s %7d %7d  %5d  %5d  %5d  %.3f  %s" % (BN[bt],bpos,bpos+blen,blen,bca,blen,bpos/44100,ks))
