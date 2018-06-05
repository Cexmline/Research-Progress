clear;close all;clc
load ('net_b3.mat');
%% Validation 
modType=4;
SymbolBits=log2(modType);
%% nonlinear parameters
M=2;
K=3;
param=[0.9 0.7 0.5 0.5 0.4 0.4 0.3 0.1 0.1];
%% NN parameters
%% modulation handles
%256QAM
hMod256 = comm.RectangularQAMModulator('ModulationOrder',256,'BitInput',true,'NormalizationMethod',...
   'Average power');
%128QAM
hMod128 = comm.RectangularQAMModulator('ModulationOrder',128,'BitInput',true,'NormalizationMethod',...
   'Average power');
%64QAM
hMod64 = comm.RectangularQAMModulator('ModulationOrder',64,'BitInput',true,'NormalizationMethod',...
   'Average power');
%32QAM
hMod32 = comm.RectangularQAMModulator('ModulationOrder',32,'BitInput',true,'NormalizationMethod',...
   'Average power');
%16QAM 
hMod16 = comm.RectangularQAMModulator('ModulationOrder',16,'BitInput',true,'NormalizationMethod',...
   'Average power');
%8QAM
hMod8 = comm.RectangularQAMModulator('ModulationOrder',8,'BitInput',true,'NormalizationMethod',...
   'Average power');
%4QAM 
hMod4 = comm.RectangularQAMModulator('ModulationOrder',4,'BitInput',true,'NormalizationMethod',...
   'Average power');
%BPSK
hMod2 = comm.BPSKModulator;

%% demodulation handles
%256QAM
hDemod256 = comm.RectangularQAMDemodulator('ModulationOrder',256,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%128QAM
hDemod128 = comm.RectangularQAMDemodulator('ModulationOrder',128,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%64QAM 
hDemod64 = comm.RectangularQAMDemodulator('ModulationOrder',64,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%32QAM 
hDemod32 = comm.RectangularQAMDemodulator('ModulationOrder',32,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%16QAM
hDemod16 = comm.RectangularQAMDemodulator('ModulationOrder',16,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%8QAM
hDemod8 = comm.RectangularQAMDemodulator('ModulationOrder',8,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%4QAM 
hDemod4 = comm.RectangularQAMDemodulator('ModulationOrder',4,'BitOutput',true,'NormalizationMethod',...
     'Average power');
%BPSK 
hDemod2 = comm.BPSKDemodulator;
switch modType
  case 256
       hMod = hMod256;
  case 128
       hMod = hMod128;
  case 64
       hMod = hMod64;
  case 32
       hMod = hMod32;  
  case 16
       hMod = hMod16;   
  case 8
       hMod = hMod8;
  case 4
       hMod = hMod4;
  case 2
       hMod = hMod2;    
  otherwise
       fprintf('error!\n');
end

switch modType
       case 256
            hDemod = hDemod256;
       case 128
            hDemod = hDemod128;
       case 64
            hDemod = hDemod64;
       case 32
            hDemod = hDemod32;   
       case 16
            hDemod = hDemod16;
       case 8
            hDemod = hDemod8;
       case 4
            hDemod = hDemod4;
       case 2
            hDemod = hDemod2; 
       otherwise
            fprintf('error!\n');
end   
%% NN parameters
input_layer_size=10;
output_layer_size=input_layer_size/2*SymbolBits;
hidden_layer_size=[128 64 32];
ClassNumber=output_layer_size; %number of outputs


nValidSymbol=2000000;
rng(112);
ValidSource=randi([0 1],nValidSymbol*SymbolBits,1);
outputcol_num=(nValidSymbol*SymbolBits)/ClassNumber;

ValidataMod=step(hMod,ValidSource);
%% nonlinear
Validxn=ValidataMod;
ValidN=length(Validxn);
ValidX=calcX(M,K,Validxn,ValidN);
Validnondata=ValidX*param';
%% add noise
ValidSNRdB=25:-1:0;
% ValidSNR=10.^(ValidSNRdB/10);
ValidationBER=zeros(1,length(ValidSNRdB));
DemodulationBER=zeros(1,length(ValidSNRdB));
ErrorMatrix=zeros(length(ValidSource),length(ValidSNRdB));
for i=1:length(ValidSNRdB)
% nGauss_Valid=wgn(length(Validnondata),1,0,'real');%%%%%%%%%%%
% Validata=Validnondata+nGauss_Valid/sqrt(ValidSNR(i));
Validata=awgn(Validnondata,ValidSNRdB(i),'measured');
Validata_vec=reshape([real(Validata)';imag(Validata)'],[],1);
%% pass NN
%% Matrix Transform
Validinput_p2=reshape(Validata_vec(2*M+1:end),input_layer_size-2*M,[]);
Validinput_p1=[Validata_vec(1:2*M)  Validinput_p2(end-M*2+1:end,1:end-1)];
Validinput=[Validinput_p1;Validinput_p2];

% Validinput=reshape(Validata_vec,input_layer_size,[]);
Validoutput= net(Validinput,'UseParallel','yes');
ThreMatrix_Valid=0.5*ones(size(Validoutput));
p_Valid=double(Validoutput>=ThreMatrix_Valid);%еп╬Ж

%% select the right part
pVec_Valid=[p_Valid(1:SymbolBits*M,1);reshape(p_Valid(SymbolBits*M+1:end,:),[],1)];


% [r,c]=size(p_Valid);
% pVec_Valid=reshape(p_Valid,r*c,1);
ErrorMatrix(:,i)=pVec_Valid-ValidSource;
ValidationBER(i)=sum(abs(pVec_Valid-ValidSource))/length(ValidSource);
fprintf('\nValidation Set BER: %f\n',ValidationBER(i));

%% demodulation directly

% demodulation
  ValidSource_rec= step(hDemod,Validata);
  nError=sum(abs(ValidSource_rec-ValidSource));
  DemodulationBER(i)=nError/length(ValidSource);
  fprintf('Demodulation BER :%f\n',  DemodulationBER(i));
end
ValidSNRdB=25:-1:0;
semilogy(ValidSNRdB,ValidationBER,'-r*','LineWidth',2);
title('k=3 m=2 QPSK');
xlabel('SNR[dB]');
ylabel('BER');
hold on;
semilogy(ValidSNRdB,DemodulationBER,'--bd','LineWidth',2);
legend('NN demodulation','QPSK demodulator');
hold off;
