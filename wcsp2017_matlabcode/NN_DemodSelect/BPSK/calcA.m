function [A,X]=calcA(M,K,xn,zn,N)
    xn=[zeros(M,1);xn];
    X=zeros(N,K*(M+1));
    head=1;

        for  n=M+1:-1:1
          Xpart=zeros(N,K);
          temp=n;    
             for i=1:N
               for j=1:K
                   Xpart(i,j)=xn(temp)*(abs(xn(temp))^(j-1));  
               end
               temp=temp+1;
             end
            X(:,head:head+K-1)=Xpart;
            head=head+K;
        end  
%       one=ones(size(X,1),1);
%       X=[one X];
      Z=zn(1:N,1);
      A=pinv(X)*Z;   
%       A=(inv(X'*X))*X'*Z;
      
end