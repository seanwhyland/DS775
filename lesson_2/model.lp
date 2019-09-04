\* Source Pyomo model name=model1 *\

min 
total_cost:
+26 daily_schedule(dh_fri)
+26 daily_schedule(dh_mon)
+26 daily_schedule(dh_thurs)
+26 daily_schedule(dh_tues)
+26 daily_schedule(dh_weds)
+24 daily_schedule(hb_fri)
+24 daily_schedule(hb_mon)
+24 daily_schedule(hb_thurs)
+24 daily_schedule(hb_tues)
+24 daily_schedule(hb_weds)
+25 daily_schedule(kc_fri)
+25 daily_schedule(kc_mon)
+25 daily_schedule(kc_thurs)
+25 daily_schedule(kc_tues)
+25 daily_schedule(kc_weds)
+28 daily_schedule(ks_fri)
+28 daily_schedule(ks_mon)
+28 daily_schedule(ks_thurs)
+28 daily_schedule(ks_tues)
+28 daily_schedule(ks_weds)
+30 daily_schedule(nk_fri)
+30 daily_schedule(nk_mon)
+30 daily_schedule(nk_thurs)
+30 daily_schedule(nk_tues)
+30 daily_schedule(nk_weds)
+23 daily_schedule(sc_fri)
+23 daily_schedule(sc_mon)
+23 daily_schedule(sc_thurs)
+23 daily_schedule(sc_tues)
+23 daily_schedule(sc_weds)

s.t.

c_l_weekly_min_hours_c(1)_:
+1 daily_schedule(kc_fri)
+1 daily_schedule(kc_mon)
+1 daily_schedule(kc_thurs)
+1 daily_schedule(kc_tues)
+1 daily_schedule(kc_weds)
>= 8

c_l_weekly_min_hours_c(2)_:
+1 daily_schedule(dh_fri)
+1 daily_schedule(dh_mon)
+1 daily_schedule(dh_thurs)
+1 daily_schedule(dh_tues)
+1 daily_schedule(dh_weds)
>= 8

c_l_weekly_min_hours_c(3)_:
+1 daily_schedule(hb_fri)
+1 daily_schedule(hb_mon)
+1 daily_schedule(hb_thurs)
+1 daily_schedule(hb_tues)
+1 daily_schedule(hb_weds)
>= 8

c_l_weekly_min_hours_c(4)_:
+1 daily_schedule(sc_fri)
+1 daily_schedule(sc_mon)
+1 daily_schedule(sc_thurs)
+1 daily_schedule(sc_tues)
+1 daily_schedule(sc_weds)
>= 8

c_l_weekly_min_hours_c(5)_:
+1 daily_schedule(ks_fri)
+1 daily_schedule(ks_mon)
+1 daily_schedule(ks_thurs)
+1 daily_schedule(ks_tues)
+1 daily_schedule(ks_weds)
>= 7

c_l_weekly_min_hours_c(6)_:
+1 daily_schedule(nk_fri)
+1 daily_schedule(nk_mon)
+1 daily_schedule(nk_thurs)
+1 daily_schedule(nk_tues)
+1 daily_schedule(nk_weds)
>= 7

c_e_weekly_max_hours_c_:
+1 daily_schedule(dh_fri)
+1 daily_schedule(dh_mon)
+1 daily_schedule(dh_thurs)
+1 daily_schedule(dh_tues)
+1 daily_schedule(dh_weds)
+1 daily_schedule(hb_fri)
+1 daily_schedule(hb_mon)
+1 daily_schedule(hb_thurs)
+1 daily_schedule(hb_tues)
+1 daily_schedule(hb_weds)
+1 daily_schedule(kc_fri)
+1 daily_schedule(kc_mon)
+1 daily_schedule(kc_thurs)
+1 daily_schedule(kc_tues)
+1 daily_schedule(kc_weds)
+1 daily_schedule(ks_fri)
+1 daily_schedule(ks_mon)
+1 daily_schedule(ks_thurs)
+1 daily_schedule(ks_tues)
+1 daily_schedule(ks_weds)
+1 daily_schedule(nk_fri)
+1 daily_schedule(nk_mon)
+1 daily_schedule(nk_thurs)
+1 daily_schedule(nk_tues)
+1 daily_schedule(nk_weds)
+1 daily_schedule(sc_fri)
+1 daily_schedule(sc_mon)
+1 daily_schedule(sc_thurs)
+1 daily_schedule(sc_tues)
+1 daily_schedule(sc_weds)
= 70

c_e_daily_max_hours_c(1)_:
+1 daily_schedule(dh_mon)
+1 daily_schedule(hb_mon)
+1 daily_schedule(kc_mon)
+1 daily_schedule(ks_mon)
+1 daily_schedule(nk_mon)
+1 daily_schedule(sc_mon)
= 14

c_e_daily_max_hours_c(2)_:
+1 daily_schedule(dh_tues)
+1 daily_schedule(hb_tues)
+1 daily_schedule(kc_tues)
+1 daily_schedule(ks_tues)
+1 daily_schedule(nk_tues)
+1 daily_schedule(sc_tues)
= 14

c_e_daily_max_hours_c(3)_:
+1 daily_schedule(dh_weds)
+1 daily_schedule(hb_weds)
+1 daily_schedule(kc_weds)
+1 daily_schedule(ks_weds)
+1 daily_schedule(nk_weds)
+1 daily_schedule(sc_weds)
= 14

c_e_daily_max_hours_c(4)_:
+1 daily_schedule(dh_thurs)
+1 daily_schedule(hb_thurs)
+1 daily_schedule(kc_thurs)
+1 daily_schedule(ks_thurs)
+1 daily_schedule(nk_thurs)
+1 daily_schedule(sc_thurs)
= 14

c_e_daily_max_hours_c(5)_:
+1 daily_schedule(dh_fri)
+1 daily_schedule(hb_fri)
+1 daily_schedule(kc_fri)
+1 daily_schedule(ks_fri)
+1 daily_schedule(nk_fri)
+1 daily_schedule(sc_fri)
= 14

c_u_daily_max_hours_per_person_c(1)_:
+1 daily_schedule(kc_mon)
<= 6

c_u_daily_max_hours_per_person_c(2)_:
+1 daily_schedule(kc_tues)
<= 0

c_u_daily_max_hours_per_person_c(3)_:
+1 daily_schedule(kc_weds)
<= 6

c_u_daily_max_hours_per_person_c(4)_:
+1 daily_schedule(kc_thurs)
<= 0

c_u_daily_max_hours_per_person_c(5)_:
+1 daily_schedule(kc_fri)
<= 6

c_u_daily_max_hours_per_person_c(6)_:
+1 daily_schedule(dh_mon)
<= 0

c_u_daily_max_hours_per_person_c(7)_:
+1 daily_schedule(dh_tues)
<= 6

c_u_daily_max_hours_per_person_c(8)_:
+1 daily_schedule(dh_weds)
<= 0

c_u_daily_max_hours_per_person_c(9)_:
+1 daily_schedule(dh_thurs)
<= 6

c_u_daily_max_hours_per_person_c(10)_:
+1 daily_schedule(dh_fri)
<= 0

c_u_daily_max_hours_per_person_c(11)_:
+1 daily_schedule(hb_mon)
<= 4

c_u_daily_max_hours_per_person_c(12)_:
+1 daily_schedule(hb_tues)
<= 8

c_u_daily_max_hours_per_person_c(13)_:
+1 daily_schedule(hb_weds)
<= 4

c_u_daily_max_hours_per_person_c(14)_:
+1 daily_schedule(hb_thurs)
<= 0

c_u_daily_max_hours_per_person_c(15)_:
+1 daily_schedule(hb_fri)
<= 4

c_u_daily_max_hours_per_person_c(16)_:
+1 daily_schedule(sc_mon)
<= 5

c_u_daily_max_hours_per_person_c(17)_:
+1 daily_schedule(sc_tues)
<= 5

c_u_daily_max_hours_per_person_c(18)_:
+1 daily_schedule(sc_weds)
<= 5

c_u_daily_max_hours_per_person_c(19)_:
+1 daily_schedule(sc_thurs)
<= 0

c_u_daily_max_hours_per_person_c(20)_:
+1 daily_schedule(sc_fri)
<= 5

c_u_daily_max_hours_per_person_c(21)_:
+1 daily_schedule(ks_mon)
<= 3

c_u_daily_max_hours_per_person_c(22)_:
+1 daily_schedule(ks_tues)
<= 0

c_u_daily_max_hours_per_person_c(23)_:
+1 daily_schedule(ks_weds)
<= 3

c_u_daily_max_hours_per_person_c(24)_:
+1 daily_schedule(ks_thurs)
<= 8

c_u_daily_max_hours_per_person_c(25)_:
+1 daily_schedule(ks_fri)
<= 0

c_u_daily_max_hours_per_person_c(26)_:
+1 daily_schedule(nk_mon)
<= 0

c_u_daily_max_hours_per_person_c(27)_:
+1 daily_schedule(nk_tues)
<= 0

c_u_daily_max_hours_per_person_c(28)_:
+1 daily_schedule(nk_weds)
<= 0

c_u_daily_max_hours_per_person_c(29)_:
+1 daily_schedule(nk_thurs)
<= 6

c_u_daily_max_hours_per_person_c(30)_:
+1 daily_schedule(nk_fri)
<= 2

c_e_ONE_VAR_CONSTANT: 
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= daily_schedule(dh_fri) <= +inf
   0 <= daily_schedule(dh_mon) <= +inf
   0 <= daily_schedule(dh_thurs) <= +inf
   0 <= daily_schedule(dh_tues) <= +inf
   0 <= daily_schedule(dh_weds) <= +inf
   0 <= daily_schedule(hb_fri) <= +inf
   0 <= daily_schedule(hb_mon) <= +inf
   0 <= daily_schedule(hb_thurs) <= +inf
   0 <= daily_schedule(hb_tues) <= +inf
   0 <= daily_schedule(hb_weds) <= +inf
   0 <= daily_schedule(kc_fri) <= +inf
   0 <= daily_schedule(kc_mon) <= +inf
   0 <= daily_schedule(kc_thurs) <= +inf
   0 <= daily_schedule(kc_tues) <= +inf
   0 <= daily_schedule(kc_weds) <= +inf
   0 <= daily_schedule(ks_fri) <= +inf
   0 <= daily_schedule(ks_mon) <= +inf
   0 <= daily_schedule(ks_thurs) <= +inf
   0 <= daily_schedule(ks_tues) <= +inf
   0 <= daily_schedule(ks_weds) <= +inf
   0 <= daily_schedule(nk_fri) <= +inf
   0 <= daily_schedule(nk_mon) <= +inf
   0 <= daily_schedule(nk_thurs) <= +inf
   0 <= daily_schedule(nk_tues) <= +inf
   0 <= daily_schedule(nk_weds) <= +inf
   0 <= daily_schedule(sc_fri) <= +inf
   0 <= daily_schedule(sc_mon) <= +inf
   0 <= daily_schedule(sc_thurs) <= +inf
   0 <= daily_schedule(sc_tues) <= +inf
   0 <= daily_schedule(sc_weds) <= +inf
end
