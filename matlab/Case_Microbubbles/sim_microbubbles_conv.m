% Function:generating random scatters for simulating microbubbles US image
% Author:Bo Peng
% Date  :Feb-5-2021
% Reference:Deep Learning for Ultrasound Localization Microscopy
% IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 39, NO. 10, OCTOBER 2020

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       no input variable below this point  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
fs = 13;            %sampling frequency in MHz????
fc  = 2;			%center frequency in MHz???????
bw = 0.8;           %bandwidth of the pulse (percent) at bwr down from peak?????
bwr = -10;          %reference level for describing pulse bandwith (dB)???????????(dB)
sigl = 0.15;        %sigma (mm) for the Gaussian lateral PSF????PSF
dr = 70;            %"dynamic range" of the system---a cutoff in pulse amplitude                  %see gauspuls calculation and help in MATLAB
a  = 0;			    %no compression
shift = 10;		    %shift in samples

xducer = 1;         % if center frequency is 5MHz
                    
c = 1.540;          %sound speed  ?mm/microsec?
depth =32;          %  
width =32;          % pulse-echo to estimate the width [mm]
FOVw = width;        %width of the image field mm
dt = 1/fs;	        %time inteval for sampling (microseconds)??????

N_lines = 320;			%number of lines for statistics??????
x_spacing = width/N_lines;  %lateral beam spacing in mm???????
y_spacing = dt*c/2; %axial spacing between samples???????
N_samples = ceil(depth/y_spacing);          %number of axial samples in the A-line ????????????

du = 1/((N_samples-1)*dt);	%frequency sample spacing (1/gate)
t2 = (N_samples)*dt/2;
u2 = 1/(2*dt);		%upper and lower frequency limit
t  = -t2:dt:t2;		%time line
u  = -u2:du:u2;		%positions of frequency samples

%SNR = 10^(SNR/20);	                            %SNR as amplitude ratio
tc = gauspuls('cutoff',fc*1e6, bw, bwr,-dr);    %determine pulse duration     tc = gauspuls('cutoff', fc*1e6, bw, bwr,-dr); 
tp  = -tc : dt/1e6 : tc;                        %calculate time during pulse

[pulseax,trash,env] = gauspuls(tp,fc*1e6,bw, bwr);  %calculate the axial pulse

dx = FOVw/N_lines;                              %lateral spatial sampling interval

x_loc  = -(N_lines-1)/2*dx:dx:N_lines/2*dx;         %index of lateral samples

pulselat = exp(-x_loc.^2/(2*sigl^2));               %lateral pusle shape

trash = find(abs(pulselat)>1/10^(dr/20));       %limits on useful lateral pulse PSF
pulselat = pulselat(min(trash):max(trash));     %lateral pulse PSF within limits

pulse = pulseax'*pulselat;                      %2-D rf pulse
% plot 2D psf
[P_R,P_C]=size(pulse);
[PX,PY] = meshgrid(1:1:P_C,1:1:P_R);
surf(PX,PY,pulse)

% pscat = randn(N_samples,N_lines);               %random 2-D distribution of point scatterers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------set up the paramters of numerical microbubbles phantom-----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size of phantom (matlab coordinate system)
x_size=32;   % [mm]
y_size=32;   % [mm]
% microbubbles density conditions
mb_density=0.32;    %0.32 - 3.2 MBs/mm-2
% number of scatters (microbubbles)
N=floor(x_size*y_size*mb_density);
% random scatters (microbubbles)
for j = 1:1000
    rng('shuffle');
x0 = rand(N,1);
x = (x0)* x_size;
y0 = rand(N,1);
y = y0*y_size;
%-------------find the index for all scatters (microbubbles)--------------%
dx=x_size/N_lines;
dy=y_size/N_samples;
xindex = round((x)/dx + 1);
yindex = round((y)/dy + 1);
amp_img=zeros(N_samples,N_lines);
for i=1:N %% number of scatters
    scat_x=xindex(i,1);
    scat_y=yindex(i,1);
    amp_img(scat_y,scat_x)=255;
end
%figure,imagesc(amp_img),colormap(gray)
 %axis image

rf1 = conv2(amp_img,pulse,'same');                %simulated 2-D rf echo field
index = find(isnan(rf1));
rf1(index) = 0; 
%figure,imagesc(rf1),colormap(gray)

bmode=bmodeConverter(rf1);
amp_img=imresize(amp_img,[320,320]);
bmode=imresize(bmode,[320,320]);

%figure,imagesc(amp_img),colormap(gray);
%figure,imagesc(bmode),colormap(gray);
bmode_capture=bmode(1:128,1:128);
%figure,imagesc(bmode_capture),colormap(gray);
table=uint8(bmode_capture);
imwrite(table,['table/table_',sprintf('%03d',j),'.png'],'png');
bmode_capture=imresize(bmode_capture,[32,32],'bicubic');
bmode_capture=uint8(bmode_capture);
imwrite(bmode_capture,['data/data_',sprintf('%03d',j),'.png'],'png');
end