# Manual benchmarking of OOS method with varying values of Psi
Same experiment and conditions, psi is the only variable. All times reported are in seconds

Run manually on 05/04/2024 (TODO: automate)


Code,Psi,Runtime1,Runtime2,Runtime3
Base LibKGE,N/A,134,139,122
OOS,0.0,128,142,121
OOS,0.1,297,221,302
OOS,0.5,861,827,671
OOS,0.9,757,833,946

As psi goes up, runtime increases a lot. Need to improve speed of the aggreation function.