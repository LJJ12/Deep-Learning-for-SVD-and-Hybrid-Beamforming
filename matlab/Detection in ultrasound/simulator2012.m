%% set up parameters
clc,clear;
c = 1540;   %speed of sound in m/s      
z_depth = 4e-2;     %m
x_width = 3e-2;     %m
Fs = 80e6;         %采样频率(Hz)
    
decimated_length = 512;  %图像的最终长度
beamlines = 300;    %最终想要的光束数
beamlines = decimated_length * x_width/z_depth;    %this it the final number of desired beamlines
      
%set up point spread function
%f1      = 7.5e6;            %中心频率(Hz)
%f_number = 1.5;
%theta_steer1 = 0 /360*2*pi;  %radians



lambda  = c/f1;             %波长（m）

t_depth = 2 * z_depth/c;    
Ts = 1/Fs;

axial_length_samples = ceil(t_depth *Fs);
axial_spacing = z_depth/axial_length_samples;

lateral_spacing = x_width/beamlines;    %lateral sampling period in meters
LATERAL_SPACING_OVER_AXIAL_SPACING_SIMULATION_RATIO = 5; %this constant is experimental
%lateral sampling also needs to be sufficiently fine in the convolution process, and we will couple this to the axial sampling.
lateral_spacing = axial_spacing * LATERAL_SPACING_OVER_AXIAL_SPACING_SIMULATION_RATIO;
simulation_beamlines = ceil(x_width/lateral_spacing) ;%number of beamlines the simulation runs on
x_sample_frequency = 1/lateral_spacing; %1/m

sigma_t1 = sqrt(2*log(2))/(pi * B1);  
sigma_f1 = 1/(2*pi*sigma_t1);

%建立高斯变迹
%use 20 db Gaussian apodization, that is, our aperture used matches
%高斯函数的全宽度最大值为0.1
beta__ = 0.1;  % 对于20db高斯变迹

alpha__ = lambda * f_number * (2*sqrt(-2*log(beta__)));
sigmax  = alpha__/(2 *sqrt(2)*pi);


t_pulse_length_threshold = 0.01;  %distance from maximum of pulse to pulse_length_threshold of the max is our definition of pulse length
t_half_pulse_length_in_samples = ceil(sqrt(  log(t_pulse_length_threshold) * -2*sigma_t1^2   ) *Fs);


x_pulse_length_threshold = 0.01; 
x_half_pulse_length_in_samples = ceil(sqrt(  log(x_pulse_length_threshold) * -2*sigmax^2   ) / lateral_spacing);



%% 创建点散射函数 (PSF)
vt1 = linspace(-t_half_pulse_length_in_samples*Ts,t_half_pulse_length_in_samples*Ts  , 2*t_half_pulse_length_in_samples);
vx1 = linspace(-x_half_pulse_length_in_samples*lateral_spacing, x_half_pulse_length_in_samples * lateral_spacing, 2*x_half_pulse_length_in_samples);
vx1 = transpose(vx1(:)); %force row vector

t_psf = exp(j*2*pi*f1* vt1).* exp(-1/(2*sigma_t1^2) * vt1.^2 );
x_psf = exp(j*2*pi* 2*tan(theta_steer1)/lambda * vx1).* exp(-1/(2*sigmax^2) * vx1.^2 );

t_x_psf = t_psf(:) * x_psf;

%-- display of the 2D PSF   
%figure; surf(real(t_x_psf))
%%  Create image with embedded object

% load image for scatterer background
%file_to_load = 'F:\mSPCN-ULM\Detection in ultrasound\background\pirate ship outline.jpg';
if exist('file_to_load')
I_background = imread(file_to_load);   %read into grayscale
I_background = mean(double(I_background),3);
subplot(121), imagesc(I_background);colormap('gray')

%dynamic_range_of_imprint = -3;  %this can be an imported variable
ratio_of_imprint = 10^(dynamic_range_of_imprint/20);

I_background_adj = double(I_background >= 128) + double(I_background < 128)*ratio_of_imprint;
% imagesc(I_background_adj,[0 1]);colormap('gray')

I_background_resized = imresize(I_background_adj, [axial_length_samples simulation_beamlines]);
% imagesc(I_background_resized,[0 1]);colormap('gray')
else
   I_background_resized = ones( axial_length_samples, simulation_beamlines);
end

%% Convolve
% 这将把超声成像作为一个线性时不变系统，这在现实生活中并不成立，但仍然是一个可接受的近似

%rand('twister',5489)  %建立一个固定的散射场
scatterer_field = rand(axial_length_samples, simulation_beamlines)/ sqrt(t_half_pulse_length_in_samples*x_half_pulse_length_in_samples); 
%对对图像有贡献的散射体数量进行归一化，这将与模拟中使用的点扩展函数的大小有关

scatterer_field = scatterer_field.*I_background_resized;

RF1 = imfilter(scatterer_field,  real(t_x_psf), 'replicate');
%在这里，我们向下采样x方向，以减少x维尺寸到最终所需的波束线的数量
RF1  = imresize(RF1,[axial_length_samples beamlines]);

% converting RF signal to its analytic form
IQ1  = hilbert(RF1);

%% Baseband and decimation
t = [0:1/Fs: (axial_length_samples-1)/Fs]';
demod_matrix = exp(-j*f1 *2*pi.*t) * ones(1,beamlines);

%demodulation
IQ1_bb = IQ1 .* demod_matrix;
%figure;imagesc(abs(fftshift(fft(IQ1_bb))))

%decimation
IQ1_bb_decimated  = imresize(IQ1_bb,[decimated_length beamlines]);

%% Display
dynamic_range = [-60 -00];
x=20*log10(abs(IQ1_bb_decimated));
bmode=bmodeConverter(x);
subplot(122),imageHandle=imagesc(bmode,dynamic_range) ;colormap('gray');
%imageHandle=imshow(20*log10(abs(IQ1_bb_decimated)),dynamic_range) ;colormap('gray');


axis([0 x_width*1e2 0 z_depth*1e2 ])
axis on
set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
xlabel('cm')
ylabel('cm')