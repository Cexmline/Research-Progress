clear;close all;clc
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



% choose modulation type
modType=2;
SymbolBits=log2(modType);%number of bits per symbol
nTrainSymbol=1.2*10^6; %number of symbols
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
%% generate the pn source bit
h = commsrc.pn('GenPoly',      [14, 13, 6, 5, 3, 1, 0], ...
              'InitialStates',  [1 1 1 1 1 1 1 1 1 1 1 1 1 1],   ...
              'CurrentStates',  [1 1 1 1 1 1 1 1 1 1 1 1 1 1],   ...
              'Mask',           [0 0 0 0 0 0 0 0 0 0 0 0 0 1],   ...
              'NumBitsOut',    log2(modType)*nTrainSymbol);
source = generate(h);

%% modulation
dataMod= step(hMod,source);
%% nonlinear
%parameters
M=2;
K=3;
N=length(dataMod);
xn=real(dataMod);
TrainX=calcX(M,K,xn,N);
param=[0.9 0.7 0.5 0.6 0.5 0.5 0.3 0.2 0.2];%
nonlineardata=TrainX*param';
%% add noise
TrainSNRdB=8;
% TrainSNR=10^(TrainSNRdB/10);
% nGauss=wgn(length(dataMod),1,0,'real');
% noise=nGauss/sqrt(TrainSNR);
% data=nonlineardata+noise;
data=awgn(real(nonlineardata),TrainSNRdB,'measured');
%% Training
%% NN parameters
input_layer_size=4;%%%%?????10
output_layer_size=input_layer_size;
hidden_layer_size=[128 64 32];%[128 64 32]
ClassNumber=output_layer_size; %number of outputs
input=reshape(data,input_layer_size,[]);
target=reshape(source,ClassNumber,[]);
nTrainSymbol=length(source)/ClassNumber;
% %归一化
% [input,inputps]=mapminmax(input);

net=feedforwardnet(hidden_layer_size,'trainscg');
net=configure(net,input,target);
% net.divideFcn = '' ;%disable validation check
%net.divideParam.trainRatio = 1;
%net.divideParam.valRatio = 0;
%net.divideParam.testRatio = 0;
% net.performParam.regularization = 0.01;
% net.trainParam.epochs=10000;
% net.trainParam.show = 10; %显示次数

net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='logsig';
net.layers{3}.transferFcn='logsig';
net.layers{4}.transferFcn='tansig';

net.inputs{1}.processFcns={'fixunknowns'};%Processes matrix rows with unknown values.
net=train(net,input,target,'UseParallel','yes');
output= net(input,'UseParallel','yes');

ThreMatrix=0.5*ones(ClassNumber,nTrainSymbol);
p=double(output>=ThreMatrix);%判决
[r,c]=size(p);
pVec=reshape(p,r*c,1);
fprintf('\nTraining Set BER: %f\n',sum(abs(pVec-source))/length(source));
perf=perform(net,target,output);%default:mse% net.performFcn='mse';
% sel = randperm(size(X, 1));
% sel = sel(1:100);
% displayData(X(sel,:));

%% Validation 
nValidSymbol=10^7+2;
rng(112);
ValidSource=randi([0 1],nValidSymbol*SymbolBits,1);
ValidataMod=step(hMod,ValidSource);
%% nonlinear
Validxn=real(ValidataMod);
ValidN=length(Validxn);
ValidX=calcX(M,K,Validxn,ValidN);
Validnondata=ValidX*param';
%% add noise
ValidSNRdB=15:-1:0;
% ValidSNR=10.^(ValidSNRdB/10);
ValidationBER=zeros(1,length(ValidSNRdB));
DemodulationBER=zeros(1,length(ValidSNRdB));
ErrorMatrix=zeros(length(ValidSource),length(ValidSNRdB));
for i=1:length(ValidSNRdB)
% nGauss_Valid=wgn(length(Validnondata),1,0,'real');%%%%%%%%%%%
% Validata=Validnondata+nGauss_Valid/sqrt(ValidSNR(i));
Validata=awgn(real(Validnondata),ValidSNRdB(i),'measured');
%% pass NN
%% Matrix Transform
Validinput_p2=reshape(Validata(M+1:end),input_layer_size-M,[]);
Validinput_p1=[Validata(1:M)  Validinput_p2(end-M+1:end,1:end-1)];
Validinput=[Validinput_p1;Validinput_p2];

% Validinput=reshape(Validata,input_layer_size,[]);
Validoutput= net(Validinput,'UseParallel','yes');
ThreMatrix_Valid=0.5*ones(size(Validoutput));
p_Valid=double(Validoutput>=ThreMatrix_Valid);%判决
%% select the right part
pVec_Valid=[p_Valid(1:SymbolBits*M,1);reshape(p_Valid(SymbolBits*M+1:end,:),[],1)];

% [r,c]=size(p_Valid);
% pVec_Valid=reshape(p_Valid,r*c,1);
ErrorMatrix(:,i)=abs(pVec_Valid-ValidSource);
ValidationBER(i)=sum(abs(pVec_Valid-ValidSource))/length(ValidSource);
fprintf('\nValidation Set BER: %f\n',ValidationBER(i));

%% demodulation directly

% demodulation
  ValidSource_rec= step(hDemod,Validata);
  nError=sum(abs(ValidSource_rec-ValidSource));
  DemodulationBER(i)=nError/length(ValidSource);
  fprintf('Demodulation BER :%f\n',  DemodulationBER(i));
end
ValidSNRdB=15:-1:0;
semilogy(ValidSNRdB,ValidationBER,'-r*','LineWidth',2);
title('BPSK K=3 M=2');
xlabel('SNR[dB]');
ylabel('BER');
hold on;
semilogy(ValidSNRdB,DemodulationBER,'--bd','LineWidth',2);
legend('NN Demodulation(b_2)','BPSK Demodulator(b_2)');
hold off;
