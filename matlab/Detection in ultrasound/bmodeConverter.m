% =========================================================================
%  File bmodeavi.mat
%  Purpose: Generate bmode avi file from rf data
%  parameters:
%    1. output
%           b_mode ---  bmode data
%           cmap -----  log compressed colormap
%    
%    2. Inoput
%           filename -  location of the rf data (i.e.
%           'E:\SI\Mayo\RF_Data\EI_RF.dat')
%           bmode ----  path and name of the bmode avi file (i.e.
%           'E:\SI\Mayo\Bmode_new\B001bmode.avi')
% =========================================================================

function [b_mode,cmap]=bmodeConverter(rfdata)


DCCancelFilter1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -4, -10, -22, -38, -52, 256, -52, -38, -22, -10, -4, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]';
DCCancelFilter2 = [0 0 0 0 0 0 0 -2 -2 -5 -8 -12 -17 -23 -28 -31 256 -31 -28 -23 -17 -12 -8 -5 -2 -2 0 0 0 0 0 0 0]';
DCCancelFilter3 = [0 0 -1 -2 -2 -3 -5 -6 -8 -10 -12 -13 -15 -16 -17 -18 256 -18 -17 -16 -15 -13 -12 -10 -8 -6 -5 -3 -2 -2 -1 0 0]';
DCCancelFilter4 = [-4 -5 -6 -6 -7 -7 -8 -8 -9 -9 -9 -10 -10 -10 -10 -10 256 -10 -10 -10 -10 -10 -9 -9 -9 -8 -8 -7 -7 -6 -6 -5 -4]';

%Log Compressed Colormap
for i=1:2^8
    logComp_Lut(i)=log10(i*6+1);
end
cm=logComp_Lut.'/max(logComp_Lut);
cmap=[cm,cm,cm];
cmap=brighten(cmap,-0.4);
cmap = double(int16(cmap*255));
last=size(rfdata,1);
%% filter out DC component
AC = conv2(rfdata, DCCancelFilter3, 'same');
%% apply hilbert transform
h = hilbert(AC);
inphase = real(h);
quad = imag(h); 
mag=sqrt(inphase.*inphase+quad.*quad);
img=mag/max(max(mag));

bmode=double(int16(img*255));

% Map data instead of colormap
b_mode=zeros(size(rfdata));
for i=1:size(bmode,1),
    for j=1:size(bmode,2),
        b_mode(i,j) = cmap(bmode(i,j)+1);
    end
end
