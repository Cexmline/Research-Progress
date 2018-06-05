function X=calcX(M,K,xn,N)
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
    
end