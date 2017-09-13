clc;clear;
load('SNR13/source.mat');
load('SNR13/x1.mat');
load('SNR13/x2.mat');
load('SNR13/x3.mat');
load('SNR13/x4.mat');
load('SNR13/x5.mat');
load('SNR13/x6.mat');
load('SNR13/x7.mat');
load('SNR13/x8.mat');
load('SNR13/y.mat');
load('H_8*4.mat')

source=(double(source))';
xk_judge=double(x1<0);
var1=sum(abs(source-xk_judge));
sernum=sum(var1>0);
ser=sernum/size(source,2);
bernum=sum(var1);
ber=bernum/numel(source);

xk_judge2=double(x2<0);
var2=sum(abs(source-xk_judge2));
sernum2=sum(var2>0);
ser2=sernum2/size(source,2);
bernum2=sum(var2);
ber2=bernum2/numel(source);

xk_judge3=double(x3<0);
var3=sum(abs(source-xk_judge3));
sernum3=sum(var3>0);
ser3_13=sernum3/size(source,2);
bernum3=sum(var3);
ber3_13=bernum3/numel(source);

xk_judge4=double(x4<0);
var4=sum(abs(source-xk_judge4));
sernum4=sum(var4>0);
ser4=sernum4/size(source,2);
bernum4=sum(var4);
ber4=bernum4/numel(source);

xk_judge5=double(x5<0);
var5=sum(abs(source-xk_judge5));
sernum5=sum(var5>0);
ser5=sernum5/size(source,2);
bernum5=sum(var5);
ber5=bernum5/numel(source);

xk_judge6=double(x6<0);
var6=sum(abs(source-xk_judge6));
sernum6=sum(var6>0);
ser6=sernum6/size(source,2);
bernum6=sum(var6);
ber6=bernum6/numel(source);

xk_judge7=double(x7<0);
var7=sum(abs(source-xk_judge7));
sernum7=sum(var7>0);
ser7=sernum7/size(source,2);
bernum7=sum(var7);
ber7=bernum7/numel(source);

xk_judge8=double(x8<0);
var8=sum(abs(source-xk_judge8));
sernum8=sum(var8>0);
ser8=sernum8/size(source,2);
bernum8=sum(var8);
ber8=bernum8/numel(source);


xwave=inv(H'*H)*H'*y';
xwave_judge=double(xwave<0);
var1_decor=sum(abs(source-xwave_judge));
sernum_decor=sum(var1_decor>0);
ser_decor=sernum_decor/size(source,2);
bernum_decor=sum(var1_decor);
ber_decor=bernum_decor/numel(source);


save('~/Research-Progress/Deep MIMO Detection Code/resnet/result/layernum8/snr13dB.mat','ber3_13','ser3_13')




