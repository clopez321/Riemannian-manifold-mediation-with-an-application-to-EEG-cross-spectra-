%%% n is the sample size, p, q are the dimensions of the 2D image, s is the
%%% dimension of the selected covariates after screening, not the original
%%% dimension
%%% totalX is n by s, totalY is a np*q matrix.
%%% The (i*p-p+1):(i*p) rows of totalY correspond to the p by q response from the ith subject
function[regularizedB,DOF]=RegMatrixCVestimationDOF(totalX,totalY,p,q,s,n,lambda) 

    %profile on
    alpha=[];
    B=[];
    B{1}=zeros(p,s*q);
    B0=zeros(p,s*q);

    %profile on
    nrm = (norm(totalX,2))^2;

    delta=n/nrm;
    % delta=1e-3;
    alpha0=0;
    alpha{1}=1;
    solution=[];
    temphvalue=[];

    t=1;

    while t==1 || t==2 || abs(temphvalue{t-1}-temphvalue{t-2})/abs(temphvalue{t-2})>10^(-4)
        if t==1
           solution{t}=B{t}+(alpha0-1)/alpha{t}*(B{t}-B0);
        end
        if t>1
           solution{t}=B{t}+(alpha{t-1}-1)/alpha{t}*(B{t}-B{t-1});
        end   
               Btemp=zeros(p,s*q);
               Atemp=solution{t}-delta*dlossmatrixdividen(totalY,totalX,solution{t},s,n);
               subAtemp=zeros(p,q,s);
               subBtemp=zeros(p,q,s);
               for j=1:s
                   subAtemp(:,:,j)=Atemp(:,(j*q-q+1):(j*q));
                   [U,S,V]=svd(subAtemp(:,:,j),0);
                   avector=(diag(S));
                   bvector=(avector-lambda*delta*ones(length(avector),1)).*(avector>lambda*delta*ones(length(avector),1));
                   if p==q
                      subBtemp(:,:,j)=U*diag(bvector)*V';
                   else
                      BS=zeros(p,q);
                      BS(1:min(p,q),1:min(p,q))=diag(bvector);
                      subBtemp(:,:,j)=U*BS*V';
                   end
               end
         Btemp=reshape(subBtemp,p,s*q);          
         temphvalue{t}=hmatrixfunctiondividen(totalY,totalX,B{t},lambda,s,n);
         B{t+1}=Btemp;
         alpha{t+1}=(1+sqrt(1+(2*alpha{t})^2))/2; 
         t=t+1;
    end  

    %DOF calculation
    Aspectrum = svd(subAtemp);
    if (p~=q)
        Aspectrum(max(p,q)) = 0;
    end
    DOF = 0;
    for i=1:nnz(bvector)
        DOF = DOF + 1 ...
            + sum(Aspectrum(i)*(Aspectrum(i)-delta*lambda) ...
            ./(Aspectrum(i)^2-[Aspectrum(1:i-1); Aspectrum(i+1:p)].^2)) ...
            + sum(Aspectrum(i)*(Aspectrum(i)-delta*lambda) ...
            ./(Aspectrum(i)^2-[Aspectrum(1:i-1); Aspectrum(i+1:q)].^2));
    end

    regularizedB=B{t};
end