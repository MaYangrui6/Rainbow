

SELECT store_sales.ss_sales_price FROM store_sales , promotion WHERE store_sales.ss_promo_sk = promotion.p_promo_sk AND store_sales.ss_hdemo_sk < 1631;


SELECT COUNT ( item.i_manufact_id ) FROM item , inventory , web_returns WHERE item.i_item_sk = web_returns.wr_item_sk AND item.i_item_sk = inventory.inv_item_sk AND inventory.inv_date_sk <= 2450815 GROUP BY web_returns.wr_reversed_charge , inventory.inv_date_sk , web_returns.wr_returned_date_sk HAVING web_returns.wr_returned_date_sk != 2452352 AND inventory.inv_date_sk != 2450815 AND web_returns.wr_reversed_charge < 12.63;


SELECT SUM ( catalog_sales.cs_ext_ship_cost ) FROM catalog_sales , customer_address WHERE customer_address.ca_address_sk = catalog_sales.cs_ship_addr_sk AND NOT customer_address.ca_address_sk = 84005;


SELECT household_demographics.hd_demo_sk FROM income_band , household_demographics , store_sales WHERE income_band.ib_income_band_sk = household_demographics.hd_income_band_sk AND household_demographics.hd_demo_sk = store_sales.ss_hdemo_sk AND NOT income_band.ib_income_band_sk >= 40 GROUP BY household_demographics.hd_demo_sk HAVING COUNT ( store_sales.ss_ext_discount_amt ) < 299.07;


SELECT MIN ( call_center.cc_mkt_id ) FROM web_page , date_dim , catalog_sales , call_center , store_sales , store , store_returns WHERE store.s_closed_date_sk = date_dim.d_date_sk AND store.s_store_sk = store_returns.sr_store_sk AND store.s_store_sk = store_sales.ss_store_sk AND date_dim.d_date_sk = call_center.cc_open_date_sk AND call_center.cc_call_center_sk = catalog_sales.cs_call_center_sk AND call_center.cc_call_center_sk > 31 AND store.s_market_id = 32 GROUP BY web_page.wp_creation_date_sk HAVING web_page.wp_creation_date_sk = 34;


SELECT time_dim.t_hour , MIN ( store_sales.ss_ext_list_price ) FROM store_sales , time_dim , web_sales WHERE time_dim.t_time_sk = web_sales.ws_sold_time_sk AND time_dim.t_time_sk = store_sales.ss_sold_time_sk AND time_dim.t_hour <= 9 GROUP BY time_dim.t_hour , web_sales.ws_ship_mode_sk , time_dim.t_hour HAVING MAX ( store_sales.ss_ext_list_price ) = 2635.92 AND web_sales.ws_ship_mode_sk != 2 AND time_dim.t_hour != 9 AND MIN ( store_sales.ss_ext_list_price ) < 3713.00;


SELECT AVG ( item.i_class_id ) FROM inventory , item WHERE item.i_item_sk = inventory.inv_item_sk AND NOT item.i_manager_id < 47 AND item.i_brand_id != 8001003 AND inventory.inv_date_sk <= 2450815 GROUP BY inventory.inv_item_sk HAVING inventory.inv_item_sk != 26647;


SELECT time_dim.t_time FROM web_site , web_sales , time_dim WHERE web_site.web_site_sk = web_sales.ws_web_site_sk AND web_sales.ws_sold_time_sk = time_dim.t_time_sk AND web_site.web_close_date_sk > 32;


SELECT warehouse.w_warehouse_sq_ft , warehouse.w_gmt_offset , SUM ( catalog_returns.cr_returning_cdemo_sk ) , SUM ( catalog_returns.cr_return_amount ) , SUM ( catalog_returns.cr_return_tax ) , AVG ( warehouse.w_warehouse_sk ) , catalog_returns.cr_returned_time_sk , catalog_returns.cr_return_amt_inc_tax , SUM ( catalog_returns.cr_returned_date_sk ) FROM catalog_returns , warehouse WHERE catalog_returns.cr_warehouse_sk = warehouse.w_warehouse_sk AND NOT catalog_returns.cr_item_sk > 75446 GROUP BY catalog_returns.cr_return_amt_inc_tax , catalog_returns.cr_returned_time_sk , warehouse.w_gmt_offset , warehouse.w_warehouse_sq_ft;


SELECT SUM ( inventory.inv_warehouse_sk ) , MAX ( inventory.inv_item_sk ) , COUNT ( item.i_brand_id ) , MAX ( item.i_category_id ) , COUNT ( item.i_current_price ) , SUM ( item.i_manufact_id ) , COUNT ( inventory.inv_quantity_on_hand ) , MAX ( item.i_item_sk ) , MAX ( inventory.inv_date_sk ) , COUNT ( item.i_wholesale_cost ) , SUM ( item.i_manager_id ) , COUNT ( item.i_class_id ) , SUM ( promotion.p_promo_sk ) , MAX ( promotion.p_response_target ) , MAX ( promotion.p_item_sk ) , MAX ( promotion.p_cost ) , SUM ( promotion.p_end_date_sk ) , COUNT ( promotion.p_start_date_sk ) FROM inventory , item , promotion WHERE inventory.inv_item_sk = item.i_item_sk AND item.i_item_sk = promotion.p_item_sk AND inventory.inv_quantity_on_hand = 456;


SELECT COUNT ( customer.c_first_shipto_date_sk ) , SUM ( household_demographics.hd_income_band_sk ) , COUNT ( household_demographics.hd_dep_count ) FROM store , store_sales , customer , catalog_sales , catalog_returns , household_demographics WHERE store_sales.ss_customer_sk = customer.c_customer_sk AND store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk AND store_sales.ss_store_sk = store.s_store_sk AND customer.c_customer_sk = catalog_returns.cr_returning_customer_sk AND customer.c_customer_sk = catalog_sales.cs_ship_customer_sk AND NOT customer.c_current_addr_sk != 10338;


SELECT AVG ( time_dim.t_second ) , MIN ( time_dim.t_minute ) , AVG ( time_dim.t_time ) , MAX ( store_sales.ss_sales_price ) , MAX ( time_dim.t_hour ) , MAX ( time_dim.t_time_sk ) , COUNT ( store_sales.ss_ext_tax ) , SUM ( store_sales.ss_wholesale_cost ) , AVG ( store_sales.ss_coupon_amt ) FROM store_sales , time_dim WHERE time_dim.t_time_sk = store_sales.ss_sold_time_sk AND store_sales.ss_item_sk <= 79541;


SELECT AVG ( inventory.inv_quantity_on_hand ) , MIN ( inventory.inv_warehouse_sk ) , MIN ( inventory.inv_date_sk ) FROM inventory , warehouse WHERE inventory.inv_warehouse_sk = warehouse.w_warehouse_sk AND NOT warehouse.w_gmt_offset < 28 GROUP BY warehouse.w_gmt_offset , inventory.inv_date_sk , inventory.inv_item_sk , warehouse.w_warehouse_sk , warehouse.w_gmt_offset HAVING warehouse.w_gmt_offset < 8 AND warehouse.w_warehouse_sk <= 36 AND inventory.inv_item_sk < 26704 AND inventory.inv_date_sk <= 2450815 AND warehouse.w_gmt_offset >= 16;


SELECT AVG ( catalog_page.cp_start_date_sk ) , AVG ( catalog_page.cp_end_date_sk ) , SUM ( promotion.p_start_date_sk ) , catalog_page.cp_catalog_page_sk FROM promotion , catalog_sales , catalog_page WHERE catalog_page.cp_catalog_page_sk = catalog_sales.cs_catalog_page_sk AND catalog_sales.cs_promo_sk = promotion.p_promo_sk AND NOT catalog_page.cp_start_date_sk < 12 GROUP BY catalog_page.cp_catalog_page_sk HAVING AVG ( catalog_page.cp_catalog_page_number ) < 26;


SELECT inventory.inv_warehouse_sk , MIN ( inventory.inv_date_sk ) FROM inventory , item WHERE inventory.inv_item_sk = item.i_item_sk AND item.i_class_id < 2 GROUP BY inventory.inv_warehouse_sk;


SELECT web_returns.wr_refunded_hdemo_sk FROM web_returns , time_dim , date_dim , catalog_returns WHERE date_dim.d_date_sk = catalog_returns.cr_returned_date_sk AND catalog_returns.cr_returned_time_sk = time_dim.t_time_sk AND date_dim.d_date_sk = web_returns.wr_returned_date_sk AND time_dim.t_minute = 24 GROUP BY catalog_returns.cr_reversed_charge , web_returns.wr_refunded_hdemo_sk HAVING catalog_returns.cr_reversed_charge < 10.85;


SELECT web_site.web_tax_percentage , customer_demographics.cd_demo_sk FROM web_site , web_sales , customer_demographics WHERE web_sales.ws_web_site_sk = web_site.web_site_sk AND web_sales.ws_ship_cdemo_sk = customer_demographics.cd_demo_sk AND NOT web_site.web_tax_percentage > 22 GROUP BY customer_demographics.cd_dep_count , customer_demographics.cd_demo_sk , web_site.web_tax_percentage HAVING AVG ( web_site.web_mkt_id ) >= 24 AND customer_demographics.cd_dep_count < 6;


SELECT catalog_sales.cs_ship_cdemo_sk FROM call_center , catalog_sales , household_demographics WHERE household_demographics.hd_demo_sk = catalog_sales.cs_ship_hdemo_sk AND catalog_sales.cs_call_center_sk = call_center.cc_call_center_sk AND household_demographics.hd_dep_count <= 12;


SELECT warehouse.w_warehouse_sq_ft , catalog_sales.cs_order_number , catalog_sales.cs_sold_date_sk , warehouse.w_gmt_offset FROM warehouse , catalog_sales , catalog_page WHERE warehouse.w_warehouse_sk = catalog_sales.cs_warehouse_sk AND catalog_sales.cs_catalog_page_sk = catalog_page.cp_catalog_page_sk AND NOT catalog_sales.cs_warehouse_sk != 3;


SELECT inventory.inv_warehouse_sk , AVG ( date_dim.d_same_day_lq ) , MAX ( inventory.inv_date_sk ) FROM date_dim , inventory WHERE inventory.inv_date_sk = date_dim.d_date_sk AND date_dim.d_same_day_lq > 2476900 GROUP BY inventory.inv_warehouse_sk;


SELECT catalog_sales.cs_ship_mode_sk FROM ship_mode , catalog_sales , date_dim , call_center WHERE call_center.cc_open_date_sk = date_dim.d_date_sk AND call_center.cc_call_center_sk = catalog_sales.cs_call_center_sk AND catalog_sales.cs_ship_mode_sk = ship_mode.sm_ship_mode_sk AND NOT ship_mode.sm_ship_mode_sk >= 46 GROUP BY catalog_sales.cs_ship_mode_sk HAVING SUM ( date_dim.d_date_sk ) <= 2464291 AND AVG ( catalog_sales.cs_ship_date_sk ) != 2450883;


SELECT date_dim.d_dow FROM store_sales , date_dim WHERE date_dim.d_date_sk = store_sales.ss_sold_date_sk AND date_dim.d_last_dom > 2454954;


SELECT customer_demographics.cd_demo_sk , COUNT ( web_sales.ws_ship_hdemo_sk ) FROM customer_demographics , web_sales WHERE customer_demographics.cd_demo_sk = web_sales.ws_ship_cdemo_sk AND customer_demographics.cd_dep_employed_count >= 0 AND customer_demographics.cd_demo_sk >= 6626 GROUP BY customer_demographics.cd_demo_sk HAVING MIN ( customer_demographics.cd_dep_college_count ) > 0;


SELECT customer.c_first_shipto_date_sk , customer_address.ca_gmt_offset , AVG ( store_sales.ss_sales_price ) , customer.c_current_hdemo_sk FROM customer_address , customer , store_sales WHERE store_sales.ss_addr_sk = customer_address.ca_address_sk AND store_sales.ss_customer_sk = customer.c_customer_sk AND NOT customer.c_current_cdemo_sk < 1403919 GROUP BY customer.c_birth_month , customer.c_current_hdemo_sk , customer_address.ca_gmt_offset , customer.c_first_shipto_date_sk HAVING AVG ( store_sales.ss_quantity ) = 12 AND customer.c_birth_month <= 2 AND MAX ( store_sales.ss_list_price ) >= 95.53;


SELECT store_sales.ss_hdemo_sk FROM store_sales , customer_demographics , household_demographics WHERE household_demographics.hd_demo_sk = store_sales.ss_hdemo_sk AND store_sales.ss_cdemo_sk = customer_demographics.cd_demo_sk AND NOT customer_demographics.cd_dep_count < 6 AND household_demographics.hd_income_band_sk <= 6 GROUP BY store_sales.ss_hdemo_sk HAVING MAX ( store_sales.ss_ext_tax ) != 49.50;


SELECT inventory.inv_date_sk , date_dim.d_dow FROM store_sales , date_dim , inventory WHERE date_dim.d_date_sk = store_sales.ss_sold_date_sk AND date_dim.d_date_sk = inventory.inv_date_sk AND NOT date_dim.d_month_seq != 149 GROUP BY date_dim.d_dow , inventory.inv_date_sk HAVING SUM ( inventory.inv_quantity_on_hand ) <= 617;


SELECT web_sales.ws_ext_wholesale_cost FROM customer , web_sales WHERE customer.c_customer_sk = web_sales.ws_ship_customer_sk AND NOT customer.c_current_cdemo_sk > 756116;


SELECT warehouse.w_warehouse_sq_ft FROM inventory , warehouse WHERE inventory.inv_warehouse_sk = warehouse.w_warehouse_sk AND NOT inventory.inv_warehouse_sk > 2 GROUP BY warehouse.w_warehouse_sq_ft , warehouse.w_warehouse_sq_ft HAVING warehouse.w_warehouse_sq_ft > 48;


SELECT catalog_returns.cr_refunded_hdemo_sk , AVG ( catalog_returns.cr_refunded_customer_sk ) FROM catalog_page , catalog_returns , item , web_sales WHERE web_sales.ws_item_sk = item.i_item_sk AND item.i_item_sk = catalog_returns.cr_item_sk AND catalog_returns.cr_catalog_page_sk = catalog_page.cp_catalog_page_sk AND NOT web_sales.ws_bill_hdemo_sk < 3515 GROUP BY catalog_returns.cr_refunded_hdemo_sk;


SELECT MAX ( inventory.inv_warehouse_sk ) , COUNT ( warehouse.w_warehouse_sk ) FROM inventory , warehouse WHERE inventory.inv_warehouse_sk = warehouse.w_warehouse_sk AND inventory.inv_warehouse_sk > 2 HAVING COUNT ( inventory.inv_date_sk ) > 2450815 AND SUM ( inventory.inv_quantity_on_hand ) < 391;


SELECT customer_demographics.cd_demo_sk FROM customer_demographics , catalog_sales WHERE catalog_sales.cs_ship_cdemo_sk = customer_demographics.cd_demo_sk AND customer_demographics.cd_dep_employed_count = 0;


SELECT AVG ( catalog_returns.cr_reversed_charge ) , customer.c_birth_day , MIN ( catalog_returns.cr_returned_time_sk ) , customer.c_customer_sk FROM catalog_sales , customer , catalog_returns WHERE customer.c_customer_sk = catalog_sales.cs_ship_customer_sk AND customer.c_customer_sk = catalog_returns.cr_returning_customer_sk AND customer.c_birth_day > 22 GROUP BY customer.c_customer_sk , customer.c_birth_day;


SELECT reason.r_reason_sk , AVG ( household_demographics.hd_dep_count ) FROM customer_demographics , store_returns , time_dim , catalog_sales , reason , household_demographics WHERE store_returns.sr_reason_sk = reason.r_reason_sk AND store_returns.sr_cdemo_sk = customer_demographics.cd_demo_sk AND store_returns.sr_return_time_sk = time_dim.t_time_sk AND customer_demographics.cd_demo_sk = catalog_sales.cs_ship_cdemo_sk AND catalog_sales.cs_ship_hdemo_sk = household_demographics.hd_demo_sk AND reason.r_reason_sk >= 21 GROUP BY reason.r_reason_sk;


SELECT AVG ( time_dim.t_time ) , AVG ( date_dim.d_year ) FROM call_center , catalog_returns , customer_demographics , time_dim , date_dim , store_sales WHERE catalog_returns.cr_returning_cdemo_sk = customer_demographics.cd_demo_sk AND customer_demographics.cd_demo_sk = store_sales.ss_cdemo_sk AND catalog_returns.cr_call_center_sk = call_center.cc_call_center_sk AND catalog_returns.cr_returned_time_sk = time_dim.t_time_sk AND catalog_returns.cr_returned_date_sk = date_dim.d_date_sk AND time_dim.t_second = 19;


SELECT MIN ( date_dim.d_dow ) FROM promotion , date_dim , catalog_page , store_sales WHERE promotion.p_start_date_sk = date_dim.d_date_sk AND promotion.p_promo_sk = store_sales.ss_promo_sk AND date_dim.d_date_sk = catalog_page.cp_start_date_sk AND NOT promotion.p_item_sk <= 29;


SELECT promotion.p_cost , MIN ( inventory.inv_quantity_on_hand ) , COUNT ( promotion.p_end_date_sk ) , SUM ( inventory.inv_item_sk ) , COUNT ( promotion.p_response_target ) , MIN ( item.i_class_id ) , MAX ( inventory.inv_warehouse_sk ) , COUNT ( inventory.inv_date_sk ) , COUNT ( item.i_manager_id ) , SUM ( promotion.p_promo_sk ) FROM item , promotion , inventory WHERE inventory.inv_item_sk = item.i_item_sk AND item.i_item_sk = promotion.p_item_sk AND inventory.inv_quantity_on_hand <= 501 GROUP BY inventory.inv_warehouse_sk , inventory.inv_quantity_on_hand , promotion.p_cost HAVING inventory.inv_quantity_on_hand != 136 AND inventory.inv_warehouse_sk < 2;


SELECT AVG ( store_sales.ss_item_sk ) FROM customer , store_sales WHERE customer.c_customer_sk = store_sales.ss_customer_sk AND NOT customer.c_current_hdemo_sk = 1793 GROUP BY customer.c_customer_sk HAVING customer.c_customer_sk > 52325 AND MIN ( customer.c_birth_day ) <= 5;


SELECT web_returns.wr_account_credit FROM time_dim , web_returns , catalog_returns WHERE time_dim.t_time_sk = catalog_returns.cr_returned_time_sk AND time_dim.t_time_sk = web_returns.wr_returned_time_sk AND NOT time_dim.t_second >= 53;


SELECT AVG ( catalog_sales.cs_ext_ship_cost ) , promotion.p_item_sk , ship_mode.sm_ship_mode_sk FROM promotion , catalog_sales , household_demographics , ship_mode WHERE catalog_sales.cs_promo_sk = promotion.p_promo_sk AND catalog_sales.cs_ship_hdemo_sk = household_demographics.hd_demo_sk AND catalog_sales.cs_ship_mode_sk = ship_mode.sm_ship_mode_sk AND promotion.p_end_date_sk > 36 GROUP BY ship_mode.sm_ship_mode_sk , promotion.p_item_sk;


SELECT AVG ( web_page.wp_image_count ) FROM web_sales , date_dim , time_dim , web_page WHERE web_page.wp_creation_date_sk = date_dim.d_date_sk AND date_dim.d_date_sk = web_sales.ws_sold_date_sk AND web_page.wp_web_page_sk >= 18 HAVING MIN ( web_page.wp_link_count ) = 22;


SELECT SUM ( web_returns.wr_return_amt_inc_tax ) , MIN ( reason.r_reason_sk ) , web_returns.wr_refunded_cdemo_sk , web_returns.wr_net_loss , MAX ( web_returns.wr_return_amt ) , SUM ( web_returns.wr_return_tax ) , COUNT ( web_returns.wr_returning_addr_sk ) , SUM ( web_returns.wr_refunded_customer_sk ) FROM reason , web_returns WHERE web_returns.wr_reason_sk = reason.r_reason_sk AND NOT reason.r_reason_sk = 16 GROUP BY web_returns.wr_net_loss , web_returns.wr_refunded_cdemo_sk;


SELECT catalog_returns.cr_returning_addr_sk , ship_mode.sm_ship_mode_sk , MAX ( catalog_returns.cr_fee ) FROM ship_mode , catalog_returns WHERE catalog_returns.cr_ship_mode_sk = ship_mode.sm_ship_mode_sk AND NOT ship_mode.sm_ship_mode_sk > 47 GROUP BY ship_mode.sm_ship_mode_sk , catalog_returns.cr_returning_addr_sk HAVING MAX ( catalog_returns.cr_call_center_sk ) <= 22;


SELECT customer_demographics.cd_purchase_estimate FROM store_sales , customer_demographics WHERE customer_demographics.cd_demo_sk = store_sales.ss_cdemo_sk AND NOT customer_demographics.cd_purchase_estimate != 8000;


SELECT warehouse.w_warehouse_sk FROM inventory , warehouse WHERE warehouse.w_warehouse_sk = inventory.inv_warehouse_sk AND NOT warehouse.w_warehouse_sk != 30;


SELECT AVG ( catalog_sales.cs_ship_date_sk ) , AVG ( catalog_sales.cs_ext_wholesale_cost ) FROM household_demographics , catalog_sales , customer_address WHERE customer_address.ca_address_sk = catalog_sales.cs_ship_addr_sk AND catalog_sales.cs_ship_hdemo_sk = household_demographics.hd_demo_sk AND household_demographics.hd_vehicle_count >= -1 HAVING MIN ( household_demographics.hd_income_band_sk ) <= 4;
