

SELECT web_site.web_close_date_sk FROM date_dim , web_site , catalog_returns , time_dim WHERE time_dim.t_time_sk = catalog_returns.cr_returned_time_sk AND catalog_returns.cr_returned_date_sk = date_dim.d_date_sk AND NOT web_site.web_company_id >= 42;


SELECT customer.c_first_shipto_date_sk , AVG ( customer.c_first_sales_date_sk ) , SUM ( catalog_returns.cr_return_quantity ) , customer.c_customer_sk , AVG ( catalog_returns.cr_store_credit ) , customer.c_current_hdemo_sk , COUNT ( catalog_returns.cr_return_amt_inc_tax ) , customer.c_current_cdemo_sk , catalog_returns.cr_warehouse_sk , MAX ( customer.c_last_review_date_sk ) , SUM ( customer.c_birth_month ) , customer.c_birth_day , catalog_returns.cr_return_ship_cost , customer.c_current_addr_sk , catalog_returns.cr_refunded_cdemo_sk , catalog_returns.cr_returning_customer_sk , MAX ( customer.c_birth_year ) , catalog_returns.cr_net_loss , catalog_returns.cr_reason_sk , AVG ( catalog_returns.cr_fee ) , COUNT ( catalog_returns.cr_refunded_cash ) , catalog_returns.cr_returning_addr_sk , SUM ( catalog_returns.cr_return_amount ) , catalog_returns.cr_refunded_customer_sk , catalog_returns.cr_return_tax , catalog_returns.cr_ship_mode_sk , COUNT ( catalog_returns.cr_order_number ) , AVG ( catalog_returns.cr_returning_cdemo_sk ) , catalog_returns.cr_item_sk , catalog_returns.cr_returned_date_sk , MIN ( catalog_returns.cr_refunded_addr_sk ) , catalog_returns.cr_refunded_hdemo_sk , catalog_returns.cr_call_center_sk FROM customer , catalog_returns WHERE catalog_returns.cr_returning_customer_sk = customer.c_customer_sk AND catalog_returns.cr_reason_sk != 41 GROUP BY catalog_returns.cr_call_center_sk , catalog_returns.cr_refunded_hdemo_sk , catalog_returns.cr_returned_date_sk , catalog_returns.cr_item_sk , catalog_returns.cr_ship_mode_sk , catalog_returns.cr_return_tax , catalog_returns.cr_refunded_customer_sk , catalog_returns.cr_returning_addr_sk , catalog_returns.cr_reason_sk , catalog_returns.cr_net_loss , catalog_returns.cr_returning_customer_sk , catalog_returns.cr_refunded_cdemo_sk , customer.c_current_addr_sk , catalog_returns.cr_return_ship_cost , customer.c_birth_day , catalog_returns.cr_warehouse_sk , customer.c_current_cdemo_sk , customer.c_current_hdemo_sk , customer.c_customer_sk , customer.c_first_shipto_date_sk;


SELECT catalog_returns.cr_warehouse_sk , MIN ( customer.c_birth_month ) , MIN ( customer.c_customer_sk ) , SUM ( customer.c_current_addr_sk ) , call_center.cc_call_center_sk , AVG ( call_center.cc_company ) , COUNT ( customer.c_first_sales_date_sk ) , customer.c_birth_day , AVG ( call_center.cc_tax_percentage ) , call_center.cc_closed_date_sk , MIN ( customer.c_last_review_date_sk ) , customer.c_first_shipto_date_sk , MIN ( catalog_returns.cr_item_sk ) , MAX ( catalog_returns.cr_net_loss ) , COUNT ( customer.c_current_cdemo_sk ) , customer.c_current_hdemo_sk , MIN ( customer.c_birth_year ) , MAX ( call_center.cc_division ) , MIN ( call_center.cc_gmt_offset ) , COUNT ( call_center.cc_employees ) , MIN ( call_center.cc_open_date_sk ) , catalog_returns.cr_refunded_hdemo_sk , SUM ( catalog_returns.cr_store_credit ) , COUNT ( call_center.cc_mkt_id ) , MAX ( call_center.cc_sq_ft ) , MIN ( catalog_returns.cr_reason_sk ) , AVG ( catalog_returns.cr_return_amount ) , AVG ( catalog_returns.cr_refunded_customer_sk ) , COUNT ( catalog_returns.cr_fee ) , SUM ( catalog_returns.cr_return_ship_cost ) , MAX ( catalog_returns.cr_reversed_charge ) , MIN ( catalog_returns.cr_return_tax ) , catalog_returns.cr_refunded_cash , MIN ( catalog_returns.cr_refunded_addr_sk ) , MAX ( catalog_returns.cr_return_amt_inc_tax ) , MAX ( catalog_returns.cr_catalog_page_sk ) , catalog_returns.cr_ship_mode_sk , AVG ( catalog_returns.cr_returning_customer_sk ) , COUNT ( catalog_returns.cr_return_quantity ) , COUNT ( catalog_returns.cr_returning_hdemo_sk ) , COUNT ( catalog_returns.cr_order_number ) , COUNT ( catalog_returns.cr_returning_cdemo_sk ) , MIN ( catalog_returns.cr_returned_date_sk ) , MAX ( catalog_returns.cr_returning_addr_sk ) , catalog_returns.cr_refunded_cdemo_sk , catalog_returns.cr_call_center_sk , SUM ( catalog_returns.cr_returned_time_sk ) FROM call_center , catalog_returns , customer WHERE call_center.cc_call_center_sk = catalog_returns.cr_call_center_sk AND catalog_returns.cr_returning_customer_sk = customer.c_customer_sk AND customer.c_customer_sk > 52320 GROUP BY catalog_returns.cr_call_center_sk , catalog_returns.cr_refunded_cdemo_sk , catalog_returns.cr_ship_mode_sk , catalog_returns.cr_refunded_cash , catalog_returns.cr_refunded_hdemo_sk , customer.c_current_hdemo_sk , customer.c_first_shipto_date_sk , call_center.cc_closed_date_sk , customer.c_birth_day , call_center.cc_call_center_sk , catalog_returns.cr_warehouse_sk;


SELECT MIN ( inventory.inv_quantity_on_hand ) , MIN ( inventory.inv_item_sk ) FROM inventory , item WHERE inventory.inv_item_sk = item.i_item_sk AND item.i_category_id >= 1;


SELECT warehouse.w_warehouse_sq_ft , warehouse.w_warehouse_sk , warehouse.w_gmt_offset , catalog_sales.cs_ship_addr_sk FROM call_center , catalog_sales , customer_demographics , warehouse WHERE call_center.cc_call_center_sk = catalog_sales.cs_call_center_sk AND catalog_sales.cs_ship_cdemo_sk = customer_demographics.cd_demo_sk AND catalog_sales.cs_warehouse_sk = warehouse.w_warehouse_sk AND NOT warehouse.w_gmt_offset >= 2;


SELECT SUM ( store_sales.ss_sold_time_sk ) FROM store_sales , customer , web_sales WHERE store_sales.ss_customer_sk = customer.c_customer_sk AND customer.c_customer_sk = web_sales.ws_ship_customer_sk AND NOT store_sales.ss_quantity >= 96;


SELECT SUM ( store_sales.ss_sold_date_sk ) , MIN ( promotion.p_end_date_sk ) , store_sales.ss_coupon_amt FROM store_sales , promotion , customer , store_returns WHERE customer.c_customer_sk = store_sales.ss_customer_sk AND customer.c_customer_sk = store_returns.sr_customer_sk AND store_sales.ss_promo_sk = promotion.p_promo_sk AND customer.c_last_review_date_sk >= 2452405 GROUP BY store_sales.ss_coupon_amt;


SELECT SUM ( date_dim.d_same_day_lq ) , AVG ( date_dim.d_first_dom ) , AVG ( catalog_sales.cs_catalog_page_sk ) , SUM ( catalog_sales.cs_ext_list_price ) , catalog_sales.cs_net_paid_inc_ship , SUM ( date_dim.d_fy_year ) , SUM ( date_dim.d_date_sk ) FROM date_dim , catalog_sales WHERE date_dim.d_date_sk = catalog_sales.cs_sold_date_sk AND NOT date_dim.d_same_day_ly <= 2429518 GROUP BY catalog_sales.cs_net_paid_inc_ship;


SELECT catalog_page.cp_end_date_sk FROM store_sales , date_dim , catalog_page WHERE date_dim.d_date_sk = store_sales.ss_sold_date_sk AND date_dim.d_date_sk = catalog_page.cp_start_date_sk AND catalog_page.cp_catalog_number <= 41;


SELECT inventory.inv_date_sk , MAX ( date_dim.d_fy_week_seq ) FROM warehouse , inventory , item , date_dim WHERE inventory.inv_warehouse_sk = warehouse.w_warehouse_sk AND inventory.inv_date_sk = date_dim.d_date_sk AND inventory.inv_item_sk = item.i_item_sk AND NOT warehouse.w_warehouse_sq_ft < 33 GROUP BY inventory.inv_date_sk;


SELECT item.i_category_id , item.i_current_price , item.i_manufact_id FROM item , inventory WHERE item.i_item_sk = inventory.inv_item_sk AND NOT item.i_category_id <= 7 GROUP BY item.i_current_price , item.i_manufact_id , item.i_current_price , item.i_category_id HAVING item.i_current_price != 0.11;


SELECT item.i_wholesale_cost FROM store_sales , item WHERE store_sales.ss_item_sk = item.i_item_sk AND NOT item.i_brand_id >= 5001001;


SELECT store_returns.sr_return_amt , time_dim.t_hour , time_dim.t_time_sk FROM time_dim , web_returns , store_returns , web_page , customer_address WHERE time_dim.t_time_sk = store_returns.sr_return_time_sk AND store_returns.sr_addr_sk = customer_address.ca_address_sk AND time_dim.t_time_sk = web_returns.wr_returned_time_sk AND web_returns.wr_web_page_sk = web_page.wp_web_page_sk AND NOT store_returns.sr_return_time_sk = 36438;


SELECT catalog_sales.cs_warehouse_sk , catalog_sales.cs_net_paid_inc_ship , catalog_sales.cs_sales_price FROM date_dim , store , catalog_sales , web_returns WHERE store.s_closed_date_sk = date_dim.d_date_sk AND date_dim.d_date_sk = catalog_sales.cs_sold_date_sk AND date_dim.d_date_sk = web_returns.wr_returned_date_sk AND NOT store.s_division_id = 37;
