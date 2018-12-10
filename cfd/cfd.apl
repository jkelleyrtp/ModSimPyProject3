kEps ← 1E¯10

⎕IO←0

⍝ Right arg: rank 1 or 2 tensor
⍝ Returns: rank 2 tensor
boxvector ← {1=⍴⍴⍵ : (1,⍴⍵)⍴⍵ ⋄ ⍵}

⍝ Right arg: rank 1 or 2 tensor
⍝ Left arg: column
⍝ Returns: extracted column
mtxcol ← {⍺⌷[1](boxvector ⍵)}


⍝ Takes in vector ⍵==[xi eta xk yk nkx nky L]
⍝ Returns [F1 F2]

⍝ goal: thicc vectorization
∇ F ← calcF in

  in ← boxvector in

  xi eta xk yk nkx nky L ← ↓⍉in ⍝ Transposes then splits matrix in order to set each destructured var to a column
  
  A ← L*2 
  B ← 2 × L × (¯nky × (xk - xi) + nkx * (yk - eta))
  E ← (xk-xi)*2 + (yk-eta)*2
  M ← (|(4×A×E - B*2))*0.5
  BA ← B ÷ A
  EA ← E ÷ A
  :If (M < kEps)
    F1 ← (0.5×÷○1) × L × (⍟L + ((1+0.5×BA)×(⍟|(1+0.5×BA))) + ¯1 + (¯0.5×BA×(⍟|0.5×BA)))
    F2 ← 0
  :Else
    F1 ← (÷○1) × 0.25 × L × ((2.0 × ((⍟L)-1)) + (¯0.5 × BA × ⍟|EA) + ((1.0 + 0.5×BA) × ⍟|(1+BA+EA)) + (M÷A) × (¯3○(((2×A)+B)÷M) - ¯3○(B÷M)))
    F2 ← (÷○1) × L × ((nkx × (xk - xi)) + (nky × (yk - eta))) ÷ M × (¯3○(((2×A)+B)÷M) - ¯3○(B÷M))
  :EndIf
  F ← ↑ F1 F2
∇

_xi ← 0.1
_eta ← 0.0
_xk ← 0.0
_yk ← 0.0
_nkx ← 0.0
_nky ← -1.0
_L ← 0.2
_A ← 0.04
_B ← -0.04
_E ← 0.01
_M ← 0.0
_BA ← -1.0
_EA ← 0.25
_F1 ← -0.10512454850632184
_F2 ← 0.0

F1 F2 ← calcF ↑ _xi _eta _xk _yk _nkx _nky _L
