%% set up system paramters
c = 1540;   %speed of sound in m/s

z_depth = 4e-2;     %m
x_width = 3e-2;     %m
Fs = 80e6;         %sampling frequency in Hz

decimated_length = 512;  %final lenghth of image
beamlines = 300;    %this it the final number of desired beamlines
beamlines = decimated_length * x_width/z_depth;    %this it the final number of desired beamlines
%---- end set up paramaters
%% first figure
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -3;  %this can be an imported variable

% set up point spread function
f1= 7.5e6;            %center frequency in Hz
B1= 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

%% second figure 
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -10;  %this can be an imported variable

% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

%% third figure 
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -0;  %this can be an imported variable

% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',sum(100*clock))  %set up speckle pattern
simulator2012

%% Increasing M

% compare this to first figure 
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -3;  %this can be an imported variable

% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012


%% Decreasing speckle size
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -5;  %this can be an imported variable


% set up point spread function
f1      = 5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

% set up point spread function
f1      = 10e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

% set up point spread function
f1      = 15e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

%% Aperture size difference
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -5;  %this can be an imported variable

% 1.9 cm aperture (3.8 cm probe, 64 channel) vs full 5 cm linear

% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 5/1.9;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012

% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012


%% Spatial compounding
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -5;  %this can be an imported variable

% 1.9 cm aperture (3.8 cm probe, 64 channel) vs full 5 cm linear

theta_steer_array = [-15 -7.5 0 7.5 15];
clear e_steer;
steer_it = 1;
for theta_steer1 = theta_steer_array
% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
%theta_steer1 = -15 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012
e_steer{steer_it} = abs(IQ1_bb_decimated);

steer_it = steer_it+1;
end
%% Display of spatial compounding
dynamic_range = [-60 -00];

e3 = (e_steer{2}+e_steer{3}+e_steer{4})/3;
e5 = (e_steer{1}+e_steer{2}+e_steer{3}+e_steer{4}+e_steer{5})/5;
figure;
imageHandle=imshow(20*log10(e3),dynamic_range) ;colormap('gray');

axis([0 x_width*1e2 0 z_depth*1e2 ])
 axis on
 set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
 set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
 xlabel('cm')
 ylabel('cm')

figure;
imageHandle=imshow(20*log10(e5),dynamic_range) ;colormap('gray');

axis([0 x_width*1e2 0 z_depth*1e2 ])
 axis on
 set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
 set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
 xlabel('cm')
 ylabel('cm')


%% Ideal compounding
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';

dynamic_range_of_imprint = -5;  %this can be an imported variable

% 1.9 cm aperture (3.8 cm probe, 64 channel) vs full 5 cm linear

clear e_cell;
for steer_it = 1:5
% set up point spread function
f1      = 7.5e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

%rand('twister',5489)  %set up speckle pattern
simulator2012
e_cell{steer_it} = abs(IQ1_bb_decimated);

steer_it = steer_it+1;
end


%% Display of ideal compounding
dynamic_range = [-60 -00];

e3 = (e_cell{1}+e_cell{2}+e_cell{3})/3;
e5 = (e_cell{1}+e_cell{2}+e_cell{3}+e_cell{4}+e_cell{5})/5;
figure;imageHandle=imshow(20*log10(e3),dynamic_range) ;colormap('gray');

axis([0 x_width*1e2 0 z_depth*1e2 ])
 axis on
 set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
 set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
 xlabel('cm')
 ylabel('cm')

figure;
imageHandle=imshow(20*log10(e5),dynamic_range) ;colormap('gray');

axis([0 x_width*1e2 0 z_depth*1e2 ])
 axis on
 set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
 set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
 xlabel('cm')
 ylabel('cm')


%% for fun
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -3;  %this can be an imported variable

% set up point spread function
f1      = 10e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012


%% Ideal compounding of cathedral
file_to_load = 'D:\F_disk_files\mSPCN-ULM\Detection in ultrasound\background\figure1.png';

dynamic_range_of_imprint = -20;  %this can be an imported variable

% 1.9 cm aperture (3.8 cm probe, 64 channel) vs full 5 cm linear

clear e_cell;
for steer_it = 1:5
% set up point spread function
f1      = 10e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
theta_steer1 = 0 /360*2*pi;  %radians

%rand('twister',5489)  %set up speckle pattern
simulator2012
e_cell{steer_it} = abs(IQ1_bb_decimated);

steer_it = steer_it+1;
end


%% Spatial compounding of cathedral
file_to_load = 'F:\mSPCN-ULM\Detection in ultrasound\background\figure1.png';
dynamic_range_of_imprint = -20;  %this can be an imported variable

% 1.9 cm aperture (3.8 cm probe, 64 channel) vs full 5 cm linear

theta_steer_array = [-15 -7.5 0 7.5 15];
clear e_steer;
steer_it = 1;
for theta_steer1 = theta_steer_array
% set up point spread function
f1      = 10e6;            %center frequency in Hz
B1      = 0.5*f1;           %bandwidth in Hz
f_number = 1.5;
%theta_steer1 = -15 /360*2*pi;  %radians

rand('twister',5489)  %set up speckle pattern
simulator2012
e_steer{steer_it} = abs(IQ1_bb_decimated);

steer_it = steer_it+1;
end

%% Display of spatial compounding cathedral weights
dynamic_range = [-60 -00];

e3 = (1.1*e_steer{2}+0.8*e_steer{3}+1.1*e_steer{4})/3;
e5 = (1.4*e_steer{1}+1*e_steer{2}+0.8*e_steer{3}+1*e_steer{4}+1.4*e_steer{5})/5;
figure;
imageHandle=imshow(20*log10(e3),dynamic_range) ;colormap('gray');

axis([0 x_width*1e2 0 z_depth*1e2 ])
 axis on
 set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
 set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
 xlabel('cm')
 ylabel('cm')

figure;
imageHandle=imshow(20*log10(e5),dynamic_range) ;colormap('gray');

axis([0 x_width*1e2 0 z_depth*1e2 ])
 axis on
 set(imageHandle,'XData',linspace(0,x_width*1e2 , beamlines))
 set(imageHandle,'YData',linspace(0,  z_depth*1e2, decimated_length))
 xlabel('cm')
 ylabel('cm')