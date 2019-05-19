clear;
load seq1
tic;
[h,w,z]=size(seq1);
G_real=seq1;
fwhmx=2;
fwhmy=2;
gx=0.9;
gy=0.9;
%initialize psf
[X, Y]=meshgrid(1:w, 1:h);
thetax=((1+gx^2-2^(2/3)*(1-gx)^2)/2/gx)^2;
dx=sqrt(thetax/(1-thetax))*fwhmx;
thetay=((1+gy^2-2^(2/3)*(1-gy)^2)/2/gy)^2;
dy=sqrt(thetay/(1-thetay))*fwhmy;
M_real=((1-gx^2)./((1+gx^2-2*gx*dx./sqrt((Y-h/2-1).^2+(X-w/2-1).^2+dx^2)).^(1.5)));
sum_M=sum(M_real(:));
M_real=M_real/sum_M;
% hessian deconvolution
[ind_x, ind_y]=meshgrid(1:z,1:h);
[ind_x1, ind_y1]=meshgrid(1:0.5:z+0.5*1, 1:h);
rho=1;
wxx=1;
wyy=1;
wxy=2;
lambda=0.5;
mu = 10000;
filename='./save.tif';
x=Bregman(G_real, M_real, mu, lambda, rho, wxx, wyy, wxy);
x_save=interp2(ind_x, ind_y, x, ind_x1, ind_y1);
imwrite(uint16(10000*(x_save-min(x_save(:)))/(max(x_save(:))-min(x_save(:)))), filename);
toc;
