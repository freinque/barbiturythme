sr = 44100
ksmps = 1
nchnls = 2

instr 1

ifrequency = p4
idb = p5
iamplitude = ampdb (idb)

asignal_l loscil iamplitude, ifrequency, 1, 60, 0

asignal_r loscil iamplitude, ifrequency, 1, 60, 0

outs asignal_l , asignal_r
endin

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
instr 2

ifrequency = p4
idb = p5
iamplitude = ampdb (idb)

asignal_l loscil iamplitude, ifrequency, 2, 60, 0

asignal_r loscil iamplitude, ifrequency, 2, 60, 0

outs asignal_l , asignal_r
endin

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
instr 3

ifrequency = p4
idb = p5
iamplitude = ampdb (idb)

asignal_l loscil iamplitude, ifrequency, 3, 60, 0

asignal_r loscil iamplitude, ifrequency, 3, 60, 0

outs asignal_l , asignal_r
endin

