# SIR likelihood and score comparison

Rows: 1584/1584; invalid rows: 0; usable reference cells: 20/33.

## Reference validity

| Case | T | Min path ESS | Fit agreement | Usable |
|---|---:|---:|---|---|
| infection09 | 10 | 1.0 | True | False |
| infection09 | 20 | 1.0 | True | False |
| infection09 | 50 | 1.0 | True | False |
| infection0999 | 10 | 23207.4 | True | True |
| infection0999 | 20 | 19729.6 | True | True |
| infection0999 | 50 | 16577.0 | True | True |
| infection1001 | 10 | 27534.4 | True | True |
| infection1001 | 20 | 18654.5 | True | True |
| infection1001 | 50 | 20155.4 | True | True |
| infection11 | 10 | 1.0 | False | False |
| infection11 | 20 | 1.0 | True | False |
| infection11 | 50 | 1.0 | True | False |
| noise08 | 10 | 15529.1 | True | True |
| noise08 | 20 | 6562.1 | True | True |
| noise08 | 50 | 2948.2 | True | True |
| noise125 | 10 | 425.2 | True | False |
| noise125 | 20 | 1355.7 | True | True |
| noise125 | 50 | 2309.0 | True | True |
| nominal | 10 | 39560.7 | True | True |
| nominal | 20 | 34775.2 | True | True |
| nominal | 50 | 29062.1 | True | True |
| recovery09 | 10 | 1.0 | True | False |
| recovery09 | 20 | 1.0 | True | False |
| recovery09 | 50 | 1.0 | True | False |
| recovery0999 | 10 | 34863.3 | True | True |
| recovery0999 | 20 | 20468.5 | True | True |
| recovery0999 | 50 | 25639.0 | True | True |
| recovery1001 | 10 | 35730.2 | True | True |
| recovery1001 | 20 | 26539.5 | True | True |
| recovery1001 | 50 | 25957.0 | True | True |
| recovery11 | 10 | 1.0 | True | False |
| recovery11 | 20 | 1.0 | True | False |
| recovery11 | 50 | 1.4 | True | False |

## Conditional accuracy

RMSE ranges span the two references. Failed reference cells have no accuracy claims.

| Case | T | Arm | Valid seeds | Log likelihood RMSE | Infection RMSE | Recovery RMSE | Noise RMSE | Mean ESS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| infection0999 | 10 | original | 8 | 0.997–0.999 | 2.23e+03–2.23e+03 | 866–866 | 923–923 | 379.7 |
| infection0999 | 10 | conditional_uniform | 8 | 0.797–0.799 | 31.6–32 | 12–12.3 | 5.04–5.05 | 472.2 |
| infection0999 | 10 | conditional_weighted | 8 | 0.779–0.779 | 32.4–34.8 | 12.5–13.6 | 4.08–4.1 | 734.6 |
| infection0999 | 10 | identity_only | 8 | 0.916–0.917 | 29.6–30.2 | 12.2–12.6 | 5.05–5.06 | 441.4 |
| infection0999 | 10 | uniform_only | 8 | 0.83–0.831 | 32.2–32.5 | 12.4–12.7 | 3.74–3.75 | 477.0 |
| infection0999 | 10 | weighted_only | 8 | 0.822–0.824 | 71.5–74.1 | 29.2–30.4 | 1.81–1.82 | 1008.0 |
| infection0999 | 20 | original | 8 | 0.918–0.92 | 2.68e+04–2.68e+04 | 9.49e+03–9.49e+03 | 2.68e+03–2.68e+03 | 597.3 |
| infection0999 | 20 | conditional_uniform | 8 | 0.82–0.821 | 32.1–34.5 | 7.05–8.13 | 4–4.01 | 651.5 |
| infection0999 | 20 | conditional_weighted | 8 | 0.606–0.607 | 28.6–37.1 | 12.9–14.3 | 4.02–4.05 | 828.2 |
| infection0999 | 20 | identity_only | 8 | 0.952–0.954 | 31–34 | 9.64–10.6 | 4.14–4.15 | 601.8 |
| infection0999 | 20 | uniform_only | 8 | 0.854–0.855 | 33.2–35.5 | 7.6–8.71 | 3.46–3.47 | 659.3 |
| infection0999 | 20 | weighted_only | 8 | 0.852–0.856 | 86.1–96.4 | 29.8–31.2 | 1.7–1.7 | 1008.0 |
| infection0999 | 50 | original | 8 | 0.88–0.881 | 5.09e+04–5.09e+04 | 1.86e+04–1.86e+04 | 7.52e+03–7.52e+03 | 796.5 |
| infection0999 | 50 | conditional_uniform | 8 | 0.757–0.757 | 32.6–34 | 6.04–6.06 | 3.91–3.95 | 816.0 |
| infection0999 | 50 | conditional_weighted | 8 | 0.568–0.568 | 30.2–35 | 11.1–11.1 | 4.98–5.06 | 911.8 |
| infection0999 | 50 | identity_only | 8 | 0.869–0.87 | 32–33.5 | 9.19–9.21 | 5.36–5.42 | 763.6 |
| infection0999 | 50 | uniform_only | 8 | 0.804–0.805 | 33.4–34.8 | 6.59–6.61 | 3.45–3.48 | 823.7 |
| infection0999 | 50 | weighted_only | 8 | 0.82–0.822 | 89.2–95 | 28.3–28.4 | 1.63–1.64 | 1008.0 |
| infection1001 | 10 | original | 8 | 0.951–0.954 | 1.89e+03–1.89e+03 | 622–623 | 694–694 | 382.2 |
| infection1001 | 10 | conditional_uniform | 8 | 0.777–0.779 | 29.7–30 | 11.4–11.9 | 4.83–4.86 | 472.9 |
| infection1001 | 10 | conditional_weighted | 8 | 0.963–0.966 | 28.5–31.7 | 11.2–12.2 | 4.72–4.77 | 731.4 |
| infection1001 | 10 | identity_only | 8 | 0.908–0.911 | 26.9–27.5 | 11.1–11.9 | 5.08–5.11 | 442.0 |
| infection1001 | 10 | uniform_only | 8 | 0.809–0.811 | 30.3–30.5 | 11.7–12.3 | 3.61–3.63 | 477.6 |
| infection1001 | 10 | weighted_only | 8 | 0.448–0.452 | 56–60.4 | 22–24.5 | 1.22–1.22 | 1008.0 |
| infection1001 | 20 | original | 8 | 0.888–0.889 | 1.78e+04–1.78e+04 | 6.64e+03–6.65e+03 | 1.47e+04–1.47e+04 | 597.4 |
| infection1001 | 20 | conditional_uniform | 8 | 0.805–0.805 | 30.5–30.6 | 5.51–6.2 | 3.95–4 | 651.7 |
| infection1001 | 20 | conditional_weighted | 8 | 0.781–0.781 | 30.9–31.4 | 7.96–8.37 | 4.41–4.48 | 826.1 |
| infection1001 | 20 | identity_only | 8 | 0.949–0.95 | 27.5–27.6 | 9.04–9.77 | 4.15–4.2 | 602.0 |
| infection1001 | 20 | uniform_only | 8 | 0.835–0.835 | 30.7–30.8 | 5.47–6.27 | 3.33–3.36 | 659.5 |
| infection1001 | 20 | weighted_only | 8 | 0.518–0.519 | 70.9–71.9 | 21.8–23 | 1.43–1.48 | 1008.0 |
| infection1001 | 50 | original | 8 | 0.849–0.85 | 3.11e+04–3.11e+04 | 1.02e+04–1.02e+04 | 2.7e+04–2.7e+04 | 796.6 |
| infection1001 | 50 | conditional_uniform | 8 | 0.744–0.745 | 31.1–31.5 | 5.29–7.74 | 3.94–3.96 | 816.2 |
| infection1001 | 50 | conditional_weighted | 8 | 0.695–0.696 | 30.1–36.7 | 7.76–9.38 | 5.42–5.45 | 911.0 |
| infection1001 | 50 | identity_only | 8 | 0.871–0.872 | 30.6–32.6 | 13.8–16.3 | 4.43–4.44 | 763.8 |
| infection1001 | 50 | uniform_only | 8 | 0.789–0.789 | 30.9–31.6 | 5.29–8.34 | 3.41–3.42 | 823.8 |
| infection1001 | 50 | weighted_only | 8 | 0.508–0.512 | 70.2–80.6 | 21.5–25.3 | 1.3–1.31 | 1008.0 |
| noise08 | 10 | original | 8 | 2.88–2.89 | 1.31e+04–1.31e+04 | 5.22e+03–5.22e+03 | 6.17e+03–6.17e+03 | 248.1 |
| noise08 | 10 | conditional_uniform | 8 | 2.08–2.08 | 48.4–50.8 | 25.8–28 | 12.1–12.1 | 317.2 |
| noise08 | 10 | conditional_weighted | 8 | 1.89–1.9 | 63.5–67.6 | 29.1–30.7 | 14.9–14.9 | 643.1 |
| noise08 | 10 | identity_only | 8 | 2.17–2.17 | 80.1–80.4 | 40–41 | 172–172 | 271.1 |
| noise08 | 10 | uniform_only | 8 | 2.17–2.17 | 51.7–54.8 | 30.4–32.3 | 27.9–27.9 | 320.2 |
| noise08 | 10 | weighted_only | 8 | 1.37–1.37 | 104–111 | 42.5–45.1 | 2.28–2.29 | 1008.0 |
| noise08 | 20 | original | 8 | 2.84–2.85 | 8.01e+05–8.01e+05 | 2.61e+05–2.61e+05 | 2.12e+05–2.12e+05 | 461.4 |
| noise08 | 20 | conditional_uniform | 8 | 1.75–1.76 | 462–465 | 210–214 | 198–198 | 499.3 |
| noise08 | 20 | conditional_weighted | 8 | 1.45–1.46 | 62.9–65.4 | 27.9–32.5 | 12.3–12.4 | 749.1 |
| noise08 | 20 | identity_only | 8 | 1.73–1.73 | 3.5e+03–3.5e+03 | 1.17e+03–1.17e+03 | 5.47e+03–5.47e+03 | 417.2 |
| noise08 | 20 | uniform_only | 8 | 1.91–1.92 | 222–226 | 105–107 | 267–267 | 510.6 |
| noise08 | 20 | weighted_only | 8 | 1.42–1.43 | 135–140 | 43.1–48.3 | 4.17–4.18 | 1008.0 |
| noise08 | 50 | original | 8 | 2.79–2.79 | 2.66e+07–2.66e+07 | 9.45e+06–9.45e+06 | 6.41e+06–6.41e+06 | 684.7 |
| noise08 | 50 | conditional_uniform | 8 | 1.62–1.62 | 5.88e+03–5.88e+03 | 3.31e+03–3.31e+03 | 2.91e+03–2.91e+03 | 697.1 |
| noise08 | 50 | conditional_weighted | 8 | 1.35–1.35 | 60.6–62.7 | 26.1–26.5 | 14.6–14.6 | 852.4 |
| noise08 | 50 | identity_only | 8 | 1.6–1.6 | 8.13e+06–8.13e+06 | 3.62e+06–3.62e+06 | 7.5e+06–7.5e+06 | 600.9 |
| noise08 | 50 | uniform_only | 8 | 1.81–1.82 | 2.13e+03–2.13e+03 | 1.28e+03–1.28e+03 | 1.76e+03–1.76e+03 | 710.9 |
| noise08 | 50 | weighted_only | 8 | 1.45–1.45 | 131–137 | 40.4–41.1 | 3.93–3.98 | 1008.0 |
| noise125 | 20 | original | 8 | 0.496–0.504 | 1.15e+03–1.15e+03 | 417–420 | 136–136 | 701.6 |
| noise125 | 20 | conditional_uniform | 8 | 0.504–0.505 | 15.7–16.3 | 2.92–9.88 | 1.13–1.17 | 771.5 |
| noise125 | 20 | conditional_weighted | 8 | 0.275–0.277 | 11.7–12.1 | 5.55–9.65 | 2.21–2.48 | 893.2 |
| noise125 | 20 | identity_only | 8 | 0.522–0.524 | 15–15.5 | 2.98–9.7 | 1.24–1.26 | 745.4 |
| noise125 | 20 | uniform_only | 8 | 0.512–0.513 | 15.7–16.3 | 2.82–9.79 | 1.18–1.23 | 775.2 |
| noise125 | 20 | weighted_only | 8 | 0.257–0.263 | 28.5–30.1 | 6.93–12 | 0.637–0.761 | 1008.0 |
| noise125 | 50 | original | 8 | 0.449–0.463 | 1.15e+03–1.15e+03 | 417–419 | 134–134 | 865.1 |
| noise125 | 50 | conditional_uniform | 8 | 0.468–0.474 | 16.4–30.7 | 2.77–8.31 | 1.08–1.1 | 891.9 |
| noise125 | 50 | conditional_weighted | 8 | 0.278–0.292 | 18.3–36.3 | 5.9–10.9 | 2.85–3.31 | 951.4 |
| noise125 | 50 | identity_only | 8 | 0.477–0.483 | 16.3–31.1 | 2.86–8.53 | 1.16–1.21 | 866.5 |
| noise125 | 50 | uniform_only | 8 | 0.481–0.486 | 16.5–30.9 | 2.7–8.36 | 1.16–1.16 | 895.4 |
| noise125 | 50 | weighted_only | 8 | 0.277–0.287 | 40.2–59.6 | 13.3–20.1 | 0.854–1.21 | 1008.0 |
| nominal | 10 | original | 8 | 0.959–0.961 | 1.29e+03–1.29e+03 | 539–539 | 347–347 | 381.9 |
| nominal | 10 | conditional_uniform | 8 | 0.787–0.788 | 30.5–30.6 | 11.6–12.1 | 4.91–4.92 | 472.6 |
| nominal | 10 | conditional_weighted | 8 | 0.83–0.832 | 38.6–39.6 | 15.8–17.1 | 4.34–4.35 | 733.8 |
| nominal | 10 | identity_only | 8 | 0.911–0.913 | 28.2–28.4 | 11.5–12.3 | 5.03–5.04 | 441.7 |
| nominal | 10 | uniform_only | 8 | 0.819–0.82 | 31.1–31.2 | 12–12.5 | 3.67–3.67 | 477.3 |
| nominal | 10 | weighted_only | 8 | 0.834–0.836 | 79.2–80.6 | 32.9–34.9 | 1.68–1.68 | 1008.0 |
| nominal | 20 | original | 8 | 0.894–0.894 | 1.14e+04–1.14e+04 | 4.87e+03–4.87e+03 | 3.49e+03–3.49e+03 | 597.8 |
| nominal | 20 | conditional_uniform | 8 | 0.812–0.812 | 31.5–31.7 | 6.92–7.39 | 3.9–3.93 | 651.6 |
| nominal | 20 | conditional_weighted | 8 | 0.687–0.687 | 41.3–42.5 | 12.9–13.3 | 3.98–4.03 | 828.2 |
| nominal | 20 | identity_only | 8 | 0.95–0.951 | 29.1–29.4 | 9.45–9.84 | 3.96–4 | 601.9 |
| nominal | 20 | uniform_only | 8 | 0.843–0.844 | 32.2–32.4 | 7.42–7.93 | 3.38–3.41 | 659.4 |
| nominal | 20 | weighted_only | 8 | 0.741–0.742 | 95.2–97.1 | 31.4–32 | 1.75–1.78 | 1008.0 |
| nominal | 50 | original | 8 | 0.853–0.854 | 1.86e+04–1.86e+04 | 7.9e+03–7.91e+03 | 9.59e+03–9.59e+03 | 796.7 |
| nominal | 50 | conditional_uniform | 8 | 0.75–0.75 | 31.9–32.8 | 6.11–7.23 | 3.85–3.9 | 816.1 |
| nominal | 50 | conditional_weighted | 8 | 0.654–0.655 | 40.9–44.4 | 12–13.1 | 4.95–5.02 | 911.9 |
| nominal | 50 | identity_only | 8 | 0.87–0.87 | 27.7–28.5 | 7.96–8.74 | 4.55–4.61 | 763.7 |
| nominal | 50 | uniform_only | 8 | 0.796–0.796 | 32.2–33.2 | 6.74–8.05 | 3.44–3.47 | 823.7 |
| nominal | 50 | weighted_only | 8 | 0.714–0.715 | 95.8–101 | 30.4–32.1 | 1.44–1.48 | 1008.0 |
| recovery0999 | 10 | original | 8 | 0.942–0.943 | 1.09e+03–1.09e+03 | 421–421 | 622–622 | 382.7 |
| recovery0999 | 10 | conditional_uniform | 8 | 0.786–0.787 | 30.3–30.3 | 11.6–12.2 | 4.83–4.85 | 472.5 |
| recovery0999 | 10 | conditional_weighted | 8 | 0.712–0.713 | 33.1–33.3 | 11.1–12.2 | 4–4.03 | 734.0 |
| recovery0999 | 10 | identity_only | 8 | 0.913–0.914 | 27.9–27.9 | 11.3–12.4 | 5.04–5.06 | 441.6 |
| recovery0999 | 10 | uniform_only | 8 | 0.817–0.818 | 30.8–30.8 | 11.9–12.6 | 3.63–3.65 | 477.3 |
| recovery0999 | 10 | weighted_only | 8 | 0.676–0.677 | 64.6–64.9 | 21.1–24.3 | 1.07–1.08 | 1008.0 |
| recovery0999 | 20 | original | 8 | 0.861–0.862 | 7.84e+03–7.84e+03 | 3.12e+03–3.12e+03 | 2.78e+03–2.78e+03 | 598.4 |
| recovery0999 | 20 | conditional_uniform | 8 | 0.811–0.811 | 31.4–31.6 | 6.29–7.12 | 3.7–3.75 | 651.6 |
| recovery0999 | 20 | conditional_weighted | 8 | 0.66–0.66 | 37.6–39.3 | 10.6–11 | 3.99–4.06 | 828.1 |
| recovery0999 | 20 | identity_only | 8 | 0.951–0.952 | 28.5–28.9 | 9.1–10 | 4–4.06 | 602.0 |
| recovery0999 | 20 | uniform_only | 8 | 0.841–0.841 | 31.9–32.2 | 6.4–7.44 | 3.38–3.42 | 659.4 |
| recovery0999 | 20 | weighted_only | 8 | 0.718–0.719 | 82.6–85.4 | 20.9–22.3 | 1.62–1.68 | 1008.0 |
| recovery0999 | 50 | original | 8 | 0.818–0.819 | 8.69e+03–8.7e+03 | 3.65e+03–3.65e+03 | 2.82e+03–2.82e+03 | 796.9 |
| recovery0999 | 50 | conditional_uniform | 8 | 0.749–0.749 | 31.7–32.9 | 5.99–7.86 | 3.75–3.78 | 816.1 |
| recovery0999 | 50 | conditional_weighted | 8 | 0.622–0.622 | 37.6–42.4 | 10.5–11.4 | 4.96–5.03 | 911.7 |
| recovery0999 | 50 | identity_only | 8 | 0.872–0.873 | 26.6–27.8 | 7.23–9.48 | 4.11–4.14 | 763.7 |
| recovery0999 | 50 | uniform_only | 8 | 0.794–0.794 | 31.8–33.3 | 5.74–8.27 | 3.37–3.4 | 823.7 |
| recovery0999 | 50 | weighted_only | 8 | 0.718–0.718 | 83.4–91 | 20–23.2 | 1.44–1.49 | 1008.0 |
| recovery1001 | 10 | original | 8 | 0.964–0.967 | 1.2e+03–1.2e+03 | 455–457 | 1.82e+03–1.82e+03 | 382.0 |
| recovery1001 | 10 | conditional_uniform | 8 | 0.788–0.79 | 30.9–31 | 11.6–12.2 | 5.01–5.02 | 472.5 |
| recovery1001 | 10 | conditional_weighted | 8 | 0.916–0.919 | 34.5–36 | 11.9–13.9 | 4.63–4.64 | 733.4 |
| recovery1001 | 10 | identity_only | 8 | 0.91–0.913 | 28.6–28.9 | 11.7–12.5 | 5.03–5.04 | 441.7 |
| recovery1001 | 10 | uniform_only | 8 | 0.821–0.823 | 31.5–31.6 | 12.1–12.6 | 3.7–3.71 | 477.3 |
| recovery1001 | 10 | weighted_only | 8 | 0.861–0.865 | 64.2–66 | 24.7–26.7 | 1.87–1.87 | 1008.0 |
| recovery1001 | 20 | original | 8 | 0.885–0.885 | 4.45e+04–4.45e+04 | 1.67e+04–1.67e+04 | 4.56e+04–4.56e+04 | 597.5 |
| recovery1001 | 20 | conditional_uniform | 8 | 0.814–0.814 | 31.6–31.8 | 7.06–7.1 | 4.06–4.09 | 651.6 |
| recovery1001 | 20 | conditional_weighted | 8 | 0.824–0.825 | 36.2–37.4 | 12.7–12.8 | 4.54–4.58 | 827.2 |
| recovery1001 | 20 | identity_only | 8 | 0.949–0.95 | 29.7–30 | 9.79–9.83 | 4.01–4.04 | 601.8 |
| recovery1001 | 20 | uniform_only | 8 | 0.846–0.846 | 32.2–32.4 | 7.49–7.53 | 3.43–3.45 | 659.4 |
| recovery1001 | 20 | weighted_only | 8 | 0.772–0.774 | 78.4–79.9 | 26.5–26.6 | 1.23–1.25 | 1008.0 |
| recovery1001 | 50 | original | 8 | 0.844–0.845 | 5.66e+03–5.66e+03 | 3.05e+03–3.05e+03 | 5.5e+03–5.5e+03 | 796.6 |
| recovery1001 | 50 | conditional_uniform | 8 | 0.751–0.752 | 31.9–32.5 | 5.85–7.05 | 3.99–4.04 | 816.1 |
| recovery1001 | 50 | conditional_weighted | 8 | 0.79–0.791 | 35.3–38.7 | 11–12.5 | 5.51–5.58 | 911.4 |
| recovery1001 | 50 | identity_only | 8 | 0.868–0.869 | 28.4–28.9 | 8.72–9.48 | 4.49–4.53 | 763.7 |
| recovery1001 | 50 | uniform_only | 8 | 0.798–0.798 | 32.1–32.8 | 6.06–7.42 | 3.48–3.51 | 823.8 |
| recovery1001 | 50 | weighted_only | 8 | 0.767–0.771 | 78.2–82.5 | 25.4–26.7 | 1.19–1.2 | 1008.0 |

## Exploratory paired findings

Requires eight valid pairs, two-reference agreement, a95% paired t interval excluding zero, and exact sign p<=0.05. No familywise claim.

- infection0999, T20, recovery_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- infection0999, T20, recovery_score: conditional_weighted has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- infection0999, T20, recovery_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- infection0999, T20, recovery_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- infection0999, T50, recovery_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- infection0999, T50, recovery_score: conditional_weighted has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- infection0999, T50, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- infection0999, T50, recovery_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- infection0999, T50, recovery_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- infection1001, T20, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- infection1001, T50, noise_score: uniform_only has supported lower squared error than the other arm in conditional_weighted vs uniform_only.
- infection1001, T50, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- noise08, T10, infection_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- noise08, T20, noise_score: conditional_weighted has supported lower squared error than the other arm in conditional_uniform vs conditional_weighted.
- noise08, T20, noise_score: weighted_only has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- noise08, T50, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- noise125, T20, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- noise125, T50, noise_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs conditional_weighted.
- noise125, T50, noise_score: identity_only has supported lower squared error than the other arm in conditional_weighted vs identity_only.
- noise125, T50, noise_score: uniform_only has supported lower squared error than the other arm in conditional_weighted vs uniform_only.
- noise125, T50, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- noise125, T50, recovery_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- nominal, T10, infection_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- nominal, T10, infection_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- nominal, T10, infection_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- nominal, T20, infection_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- nominal, T20, recovery_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- nominal, T20, infection_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- nominal, T20, infection_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- nominal, T20, recovery_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- nominal, T50, infection_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- nominal, T50, recovery_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- nominal, T50, infection_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- nominal, T50, infection_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- nominal, T50, recovery_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- recovery0999, T10, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- recovery0999, T50, noise_score: uniform_only has supported lower squared error than the other arm in conditional_weighted vs uniform_only.
- recovery0999, T50, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- recovery1001, T20, recovery_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- recovery1001, T20, recovery_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- recovery1001, T20, recovery_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.
- recovery1001, T50, recovery_score: conditional_uniform has supported lower squared error than the other arm in conditional_uniform vs weighted_only.
- recovery1001, T50, noise_score: weighted_only has supported lower squared error than the other arm in conditional_weighted vs weighted_only.
- recovery1001, T50, recovery_score: identity_only has supported lower squared error than the other arm in identity_only vs weighted_only.
- recovery1001, T50, recovery_score: uniform_only has supported lower squared error than the other arm in uniform_only vs weighted_only.

Heuristic-dominance verdict: promotion_veto. Every conditional check is retained in summary.json.
