clc;clear
% N=8;
% load ('Hn_8x4/Hn1_8x4_8.8505.mat')
% load ('R_8x8.mat')
% HR1=R_8x8 * Hn1;
% [U,S,V]=svd(HR1);
% landa=max(diag(S))/min(diag(S));
% HRn1=(sqrt(N)/sqrt(trace(HR1'*HR1)))*HR1;

N=16;
load ('Hn_16x8/Hn3_16x8_7.2562.mat')
load ('R_16x16.mat')
HR3=R_16x16 * Hn3;
[U,S,V]=svd(HR3);
landa=max(diag(S))/min(diag(S));
HRn3=(sqrt(N)/sqrt(trace(HR3'*HR3)))*HR3;