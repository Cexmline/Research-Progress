clc;clear;
load('layernum20/16x16/simplify/ber.mat')
load('layernum20/16x16/simplify/berdecorSet.mat')

SNR=8:15;
semilogy(SNR,ber(5,:),'-c+','LineWidth',1.6);
grid on;
title('VC-simplify-16x16-QPSK-layernum20');
xlabel('SNR[dB]');
ylabel('BER');
hold on;
% semilogy(SNR,ber(2,:),'-bo','LineWidth',1.6);
semilogy(SNR,berdecorSet,'-m>','LineWidth',1.6);
legend("layer20output","decorrelation")
hold off;