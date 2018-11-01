\ Model matlab
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 20 x1 - 0.04 x2 - 0.1 x3 + 0.05 x4 + [ 1080 x1 ^2 - 12 x1 * x2
   - 0.8 x1 * x3 + 240 x2 ^2 + 20 x2 * x3 + 40 x2 * x4 + 80 x3 ^2
   + 12 x3 * x4 + 0.4 x4 ^2 ] / 2 
Subject To
 R0: x1 + x2 + x3 + x4 = 1
Bounds
 0.01 <= x1 <= 0.5
 x2 <= 1
 0.005 <= x3 <= 1
 0.03 <= x4 <= 0.4
End
