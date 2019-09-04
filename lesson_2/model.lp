\* Source Pyomo model name=model2 *\

min 
total_cost:
+170 periods_shifts(p10_12_s1)
+160 periods_shifts(p10_12_s2)
+175 periods_shifts(p10_12_s3)
+180 periods_shifts(p10_12_s4)
+195 periods_shifts(p10_12_s5)
+170 periods_shifts(p12_14_s1)
+160 periods_shifts(p12_14_s2)
+175 periods_shifts(p12_14_s3)
+180 periods_shifts(p12_14_s4)
+195 periods_shifts(p12_14_s5)
+170 periods_shifts(p14_16_s1)
+160 periods_shifts(p14_16_s2)
+175 periods_shifts(p14_16_s3)
+180 periods_shifts(p14_16_s4)
+195 periods_shifts(p14_16_s5)
+170 periods_shifts(p16_18_s1)
+160 periods_shifts(p16_18_s2)
+175 periods_shifts(p16_18_s3)
+180 periods_shifts(p16_18_s4)
+195 periods_shifts(p16_18_s5)
+170 periods_shifts(p18_20_s1)
+160 periods_shifts(p18_20_s2)
+175 periods_shifts(p18_20_s3)
+180 periods_shifts(p18_20_s4)
+195 periods_shifts(p18_20_s5)
+170 periods_shifts(p20_22_s1)
+160 periods_shifts(p20_22_s2)
+175 periods_shifts(p20_22_s3)
+180 periods_shifts(p20_22_s4)
+195 periods_shifts(p20_22_s5)
+170 periods_shifts(p22_24_s1)
+160 periods_shifts(p22_24_s2)
+175 periods_shifts(p22_24_s3)
+180 periods_shifts(p22_24_s4)
+195 periods_shifts(p22_24_s5)
+170 periods_shifts(p24_6_s1)
+160 periods_shifts(p24_6_s2)
+175 periods_shifts(p24_6_s3)
+180 periods_shifts(p24_6_s4)
+195 periods_shifts(p24_6_s5)
+170 periods_shifts(p6_8_s1)
+160 periods_shifts(p6_8_s2)
+175 periods_shifts(p6_8_s3)
+180 periods_shifts(p6_8_s4)
+195 periods_shifts(p6_8_s5)
+170 periods_shifts(p8_10_s1)
+160 periods_shifts(p8_10_s2)
+175 periods_shifts(p8_10_s3)
+180 periods_shifts(p8_10_s4)
+195 periods_shifts(p8_10_s5)

s.t.

c_l_min_agents_per_period(1)_:
+1 periods_shifts(p6_8_s1)
>= 48

c_l_min_agents_per_period(2)_:
+1 periods_shifts(p8_10_s1)
+1 periods_shifts(p8_10_s2)
>= 79

c_l_min_agents_per_period(3)_:
+1 periods_shifts(p10_12_s1)
+1 periods_shifts(p10_12_s2)
>= 65

c_l_min_agents_per_period(4)_:
+1 periods_shifts(p12_14_s1)
+1 periods_shifts(p12_14_s2)
+1 periods_shifts(p12_14_s3)
>= 87

c_l_min_agents_per_period(5)_:
+1 periods_shifts(p14_16_s2)
+1 periods_shifts(p14_16_s3)
>= 64

c_l_min_agents_per_period(6)_:
+1 periods_shifts(p16_18_s3)
+1 periods_shifts(p16_18_s4)
>= 73

c_l_min_agents_per_period(7)_:
+1 periods_shifts(p18_20_s3)
+1 periods_shifts(p18_20_s4)
>= 82

c_l_min_agents_per_period(8)_:
+1 periods_shifts(p20_22_s4)
>= 43

c_l_min_agents_per_period(9)_:
+1 periods_shifts(p22_24_s4)
+1 periods_shifts(p22_24_s5)
>= 52

c_l_min_agents_per_period(10)_:
+1 periods_shifts(p24_6_s5)
>= 15

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= periods_shifts(p10_12_s1) <= +inf
   0 <= periods_shifts(p10_12_s2) <= +inf
   0 <= periods_shifts(p10_12_s3) <= +inf
   0 <= periods_shifts(p10_12_s4) <= +inf
   0 <= periods_shifts(p10_12_s5) <= +inf
   0 <= periods_shifts(p12_14_s1) <= +inf
   0 <= periods_shifts(p12_14_s2) <= +inf
   0 <= periods_shifts(p12_14_s3) <= +inf
   0 <= periods_shifts(p12_14_s4) <= +inf
   0 <= periods_shifts(p12_14_s5) <= +inf
   0 <= periods_shifts(p14_16_s1) <= +inf
   0 <= periods_shifts(p14_16_s2) <= +inf
   0 <= periods_shifts(p14_16_s3) <= +inf
   0 <= periods_shifts(p14_16_s4) <= +inf
   0 <= periods_shifts(p14_16_s5) <= +inf
   0 <= periods_shifts(p16_18_s1) <= +inf
   0 <= periods_shifts(p16_18_s2) <= +inf
   0 <= periods_shifts(p16_18_s3) <= +inf
   0 <= periods_shifts(p16_18_s4) <= +inf
   0 <= periods_shifts(p16_18_s5) <= +inf
   0 <= periods_shifts(p18_20_s1) <= +inf
   0 <= periods_shifts(p18_20_s2) <= +inf
   0 <= periods_shifts(p18_20_s3) <= +inf
   0 <= periods_shifts(p18_20_s4) <= +inf
   0 <= periods_shifts(p18_20_s5) <= +inf
   0 <= periods_shifts(p20_22_s1) <= +inf
   0 <= periods_shifts(p20_22_s2) <= +inf
   0 <= periods_shifts(p20_22_s3) <= +inf
   0 <= periods_shifts(p20_22_s4) <= +inf
   0 <= periods_shifts(p20_22_s5) <= +inf
   0 <= periods_shifts(p22_24_s1) <= +inf
   0 <= periods_shifts(p22_24_s2) <= +inf
   0 <= periods_shifts(p22_24_s3) <= +inf
   0 <= periods_shifts(p22_24_s4) <= +inf
   0 <= periods_shifts(p22_24_s5) <= +inf
   0 <= periods_shifts(p24_6_s1) <= +inf
   0 <= periods_shifts(p24_6_s2) <= +inf
   0 <= periods_shifts(p24_6_s3) <= +inf
   0 <= periods_shifts(p24_6_s4) <= +inf
   0 <= periods_shifts(p24_6_s5) <= +inf
   0 <= periods_shifts(p6_8_s1) <= +inf
   0 <= periods_shifts(p6_8_s2) <= +inf
   0 <= periods_shifts(p6_8_s3) <= +inf
   0 <= periods_shifts(p6_8_s4) <= +inf
   0 <= periods_shifts(p6_8_s5) <= +inf
   0 <= periods_shifts(p8_10_s1) <= +inf
   0 <= periods_shifts(p8_10_s2) <= +inf
   0 <= periods_shifts(p8_10_s3) <= +inf
   0 <= periods_shifts(p8_10_s4) <= +inf
   0 <= periods_shifts(p8_10_s5) <= +inf
end
