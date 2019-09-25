\* Source Pyomo model name=max_airfare *\

max 
airfare:
+22.590018622525943 x(coupon)
+0.083335860847871618 x(distance)
+0.011797683308101192 x(hi)

s.t.

c_u_pax_:
+10819.328521572506 x(coupon)
-2.2980169884399482 x(distance)
+0.24818291567558617 x(hi)
<= 20000

c_u_s_income_:
+20909.191409050938 x(coupon)
-2.8309828112925359 x(distance)
+1.1145829662244195 x(hi)
<= 30000

c_l_e_income_:
+18330.370962437933 x(coupon)
-1.0198022655615568 x(distance)
+1.4068818371080081 x(hi)
>= 30000

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= x(coupon) <= 1.5
   500 <= x(distance) <= 1000
   4000 <= x(hi) <= 8000
end
