function x=Bregman(y, h, mu, lambda, rho, wxx, wyy, wxy)
%% initialization
iter_Bregman = 100;     %number of iteration
tao=2;
nu=10;

ymax=max(y(:));
y=y./ymax;
[sx, sy, sz] = size(y);
y1=y(:,end/2+1:end,:);
y2=y(:,1:end/2,:);
y(:,1:end/2,:)=y1;
y(:,end/2+1:end,:)=y2;
clear y1 y2 
sizex=[sx,sz];
%FFT of psf
fft_psf=fftn(ifftshift(h));
hsum=sum(fft_psf.*conj(fft_psf), 2);
hsum=repmat(hsum, 1, sz);

w_xx=wxx;
w_yy=wyy;
w_xy=wxy;

%FFT of difference operator
fft_yy=fftn([1 -2 1],sizex);
Frefft = w_yy^2 * fft_yy.*conj(fft_yy);
fft_xx=fftn([1 ;-2 ;1],sizex);
Frefft=Frefft + w_xx^2*fft_xx.*conj(fft_xx);
fft_xy=fftn([1 -1;-1 1],sizex);
Frefft=Frefft + w_xy^2 * 4 * fft_xy.*conj(fft_xy);
%FFT of g
fft_y=fftn(y);
fft_g=fft_y.*conj(repmat(fft_psf, 1, 1, sz));
fft_g=squeeze(sum(fft_g, 2)); 

%% iteration
u1 = zeros(sizex,'double');
u2 = zeros(sizex,'double');
u3 = zeros(sizex,'double');
b1 = zeros(sizex,'double');
b2 = zeros(sizex,'double');
b3 = zeros(sizex,'double');
x = zeros(sizex,'double');

for ii = 1:iter_Bregman
    %% renew x
    frac=(rho/mu*sy)*(w_xx*conj(fft_xx).*fftn(b1-u1)+w_yy*conj(fft_yy).*fftn(b2-u2)+w_xy*2*conj(fft_xy).*fftn(b3-u3))+fft_g;
    divide=(rho*sy/mu)*Frefft+hsum;
    if ii>1
        x = real(ifftn(frac./divide));
        x(x<0)=0;
    else
        x = real(ifftn(frac./divide));
        x(x<0)=0;
    end
    %% renew d
    
    u = w_xx*back_diff(back_diff(x,1,1),1,1);
    signd = abs(u+u1)-2*lambda/rho;
    signd(signd<0)=0;
    signd=signd.*sign(u+u1);
    b1_dif=signd-b1;
    b1=signd;
    u1 = u1+(u-b1);
    frac = w_xx * forward_diff(forward_diff(b1-u1,1,1),1,1);
    
    u = w_yy * back_diff(back_diff(x,1,2),1,2);
    signd = abs(u+u2)-2*lambda/rho;
    signd(signd<0)=0;
    signd=signd.*sign(u+u2);
    b2_dif=signd-b2;
    b2=signd;
    u2 = u2+(u-b2);
    frac = frac+ w_yy * forward_diff(forward_diff(b2-u2,1,2),1,2);
    
    u = w_xy * 2 * back_diff(back_diff(x,1,1),1,2);
    signd = abs(u+u3)-2*lambda/rho;%*abs(u); 
    signd(signd<0)=0;
    signd=signd.*sign(u+u3);
    b3_dif=signd-b3;
    b3=signd;
    u3 = u3+(u-b3);
    frac = frac+ w_xy * 2 * forward_diff(forward_diff(b3-u3,1,2),1,1);


    square_cost=sum(sum(sum((abs(fft_y-repmat(reshape(fftn(x),[sx, 1, sz]),1, sy, 1)/sy.*repmat(reshape(fft_psf,[sx, sy, 1]),1, 1, sz))).^2)))/(sx*sy*sz);
    hessian_cost_xx=sum(sum(abs(back_diff(back_diff(x, 1, 1), 1, 1))));
    hessian_cost_yy=sum(sum(abs(back_diff(back_diff(x, 1, 2), 1, 2))));
    hessian_cost_xy=sum(sum(abs(back_diff(back_diff(x, 1, 1), 1, 2))));
    hessian_cost=hessian_cost_xx+hessian_cost_yy+2*hessian_cost_xy;
    r=sqrt(sum(sum((abs(b1-back_diff(back_diff(x, 1, 1), 1, 1))).^2 + (abs(b2-back_diff(back_diff(x, 1, 2), 1, 2))).^2 +...
        (abs(b3-2*back_diff(back_diff(x, 1, 1), 1, 2))).^2)));
    s=rho*sqrt(sum(sum((abs(conj(fft_xx).*fftn(b1_dif))).^2+(abs(conj(fft_yy).*fftn(b2_dif))).^2+...
        (abs(2*conj(fft_xy).*fftn(b3_dif))).^2))/(sx*sz));


    
    if(r>(nu*s))
        rho=rho*tao;
    else
        if(s>(nu*r))
            rho=rho/tao;
        end
    end
    
    disp(['square loss: ', num2str(square_cost),' hessian loss: ', num2str(hessian_cost), ' prime_loss: ', num2str(r), ...
        ' dual_loss: ', num2str(s), ' rho: ', num2str(rho),' hessiancost: xx: ', num2str(hessian_cost_xx),...
        ', yy: ',num2str(hessian_cost_yy),', xy: ',num2str(2*hessian_cost_xy)]);
end
x=x.*ymax;
end
