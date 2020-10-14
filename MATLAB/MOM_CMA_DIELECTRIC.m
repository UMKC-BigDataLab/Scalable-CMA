clc
clear
clear all
   
%% mode extract
%CASE DIELECTRIC
% MAXIMUM POSSIBLE MODES for Dielectric=lz/2

num_modes=lz/2;          % request large number of modes to avoid missing mode
IC3D = cell(length(FREQ),1);

for ifa=1           % for each frequency we need to calculate
    ifa                             %display which frequency is being solved
    a=importdata(['freq' num2str(ifa)],' ',8);  % after skipping first 8 lines
    a2=complex(a.data(:,3),a.data(:,4));
    [A,B]=size(a2);
    
    b=reshape(a2,lz*lz,length(a.data)/(lz*lz));   
    ZZ=reshape(b,lz,lz);
    
%CASE Dielectric 
    % If object is dielctric then it will have same no. of electric and magnetic edges, then
    % lz=2 * edge no= matrix diemntion of freq file= no. of RWG basis functions   
    % partitioning of the ZZ matrix in 4 quadrant is needed
 
    ZEE=ZZ(1:lz/2,1:lz/2);
    ZEH=ZZ(1:lz/2,lz/2+1:end);
    ZHE=ZZ(lz/2+1:end,1:lz/2);
    ZHH=ZZ(lz/2+1:end,lz/2+1:end);
    clear ZZ
    ZZ=ZEE-ZEH*inv(ZHH)*ZHE;
 
    RR=real(ZZ);
    XX=imag(ZZ);
    
    [UU,SS,VV]=svd(RR);
    vs=diag(SS);
    
   
It=find(vs<vs(1)/1000); %find low magnitude entries along diagonal that can be discarded

if(isempty(It))         % if there is no lower order entry  
    si=length(SS);      % then keep the matrix size same
else
    si=It(1);    % else reduce the matrix size,
end  
    
    u11=SS(1:si,1:si);
    
    
    for i=1:si
        u11(i,i)=u11(i,i)^(-0.5);
    end
    
    A=transpose(UU)*XX*UU;
    A11=A(1:si,1:si);
    A12=A(1:si,si+1:end);
    A21=A(si+1:end,1:si);
    A22=A(si+1:end,si+1:end);
    
    B=(u11)*(A11-A12*inv(A22)*transpose(A12))*(u11);
    
    [UB,SB,HB]=svd(B);
    
    VB=UU*[eye(si);-inv(A22)*transpose(A12)]*u11*HB;
    
    
    for jm=1:num_modes
        IC(1:si,jm)=VB(:,si-jm+1);
        DD(jm)=imag(transpose(IC(:,jm))*ZZ*IC(:,jm));
        
    end

 %% MODE TRACKING
 
if(ifa>1)
        old_modes=load(['modes' num2str(ifa-1)]);
        old_msn=load(['MSn' num2str(ifa-1)]);
        
        for im=1:num_modes
            for jm=1:num_modes
                CM(im,jm)=abs(transpose(old_modes(:,im))*RR*IC(:,jm));
            end
            [temp,It]=max(CM(im,:));
            if(It~=im)
                temp2=DD(It);
                DD(It)=DD(im);
                DD(im)=temp2;
                temp3=IC(:,It);
                IC(:,It)=IC(:,im);
                IC(:,im)=temp3;
            end
        end
 end
 %%   Saving eigen vectors after mode tracking
 IC3D{ifa,1}= IC(:,:);      % all mode current pattern for this perticular frequency

 
 
    name_1=['modes' num2str(ifa)];
    fid=fopen(name_1,'w');
    
    for ic=1:si
        for jm=1:num_modes
            fprintf(fid,'%12.9e   ',real(IC(ic,jm)));
            %     fprintf(fid,'%s %12.9f  %12.9f %12.9f\n','   vertex', x01_1,y01_1,z01_1);
        end
        
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    name_1=['MSn' num2str(ifa)];
    fid=fopen(name_1,'w');
    
    
    for jm=1:num_modes
        fprintf(fid,'%12.9e   ',DD(jm));
        %     fprintf(fid,'%s %12.9f  %12.9f %12.9f\n','   vertex', x01_1,y01_1,z01_1);
    end
    
    fprintf(fid,'\n');
    fclose(fid);
 
end

name_1=['MSn_total_' num2str(num_modes)];
fid=fopen(name_1,'w');
for i=1:length(FREQ)
    tem=importdata(['MSn' num2str(i)]);
    for jm=1:num_modes
        DM(jm,i)=tem(jm);
        fprintf(fid,'%12.9e   ',DM(jm,i));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'\n');



 


