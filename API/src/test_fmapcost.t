./o.t:1844:             terra cost($idx : Index,$P : ProblemParameters) : float
./o.t:1845:                 var $midx : Index = $idx
./o.t:1808:                 var $r1 : float_2
                            var $r2 : float
                            var $r3 : float
                            var $r4 : float_2
                            var $r5 : float
                            var $r6 : float
                            var $r7 : float
                            var $r8 : float
                            var $r9 : float
                            var $r10 : float
                            var $r11 : float
                            var $r12 : float
                            var $r13 : float
                            var $r14 : float
./o.t:1694:                 var $A_(0,0) : float_2 = Image.metamethods.__apply(&$P.A, Index.metamethods.__apply(&$midx, 0, 0))
./o.t:1811:                 $r1 = $A_(0,0)
                            $r2 = [&float]($r1.data)[1]
                            $r3 = [float](-1) * $r2
./o.t:1694:                 var $X_(0,0) : float_2 = Image.metamethods.__apply(&$P.X.X, Index.metamethods.__apply(&$midx, 0, 0))
./o.t:1811:                 $r4 = $X_(0,0)
                            $r5 = [&float]($r4.data)[1]
                            $r6 = [float](0) + $r5 + $r3
                            $r7 = [&float]($r1.data)[0]
                            $r8 = [float](-1) * $r7
                            $r9 = [&float]($r4.data)[0]
                            $r10 = [float](0) + $r9 + $r8
                            $r11 = pow2($r6)
                            $r12 = pow2($r10)
                            $r13 = [float](0) + $r12 + $r11
                            $r14 = [float](0.5 * [double]($r13))
./o.t:1849:                 return $r14
./o.t:1844:             end

