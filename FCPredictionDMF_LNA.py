#@mfunction("fitting")
def FC_prediction_DMF_LNA(C=None, FCemp=None, Gs=None):

    #
    # Calculate moments
    #
    # Each node contains a reduced model � la Wang of excitatory and inhibitory
    # units.
    # Nodes are connected via a DTI matrix.
    #
    #
    #
    #--------------------------------------------------------------------------

    # clear all;
    #
    # # Load data:
    # #-----------
    #
    # load Human_66.mat C Order
    # load Corbetta.mat Cb Pv
    # FC_emp = Cb;
    # C=C(Order,Order);


    M  = size(C,1);

    Isubdiag = find(tril(ones(M),-1)); # Values below the diagonal.
    fc       = FCemp(Isubdiag); # Vector of all FC values below the diagonal
    Indices  = Isubdiag;

    fc_z=atanh(fc);



    #--------------------------------------------------------------------------
    # Fixed parameters and simulation length:
    #----------------------------------------

    w=1.4; # local recurrence

    tau_exc=100;
    tau_inh=10;

    gamma=0.641;
    JN=0.15;

    Jo=ones(M,1);

    I0=0.382;
    Jexte=1;
    Jexti=0.7;

    I_exc = I0*Jexte;
    I_inh = I0*Jexti;

    Io = [I_exc*ones(M,1); I_inh*ones(M,1)];

    # transfert function: excitatory
    #--------------------------------------------------------------------------
    a=310;
    b=125;
    d=0.16;
    He=@(x) (a*x-b)./(1-exp(-d*(a*x-b)));

    # it's derivative:

    H_1e=@(x) a*( 1+ (-d*(a*x-b)-1).*exp(-d*(a*x-b)) )./( (1-exp(-d*(a*x-b))).^2 );

    # transfert function: inhibitory
    #--------------------------------------------------------------------------
    a=615;
    b=177;
    d=0.087;
    Hi=@(x) (a*x-b)./(1-exp(-d*(a*x-b)));

    # it's derivative:

    H_1i=@(x) a*( 1+ (-d*(a*x-b)-1).*exp(-d*(a*x-b)) )./( (1-exp(-d*(a*x-b))).^2 );


    # simulation length and binsize:

    dt=.1;
    tmax=2000; #3000; #1000
    tspan=0:dt:tmax;


    # Noise level:
    #--------------------

    beta=0.01;  # additive ("finite size") noise

    #--------------------------------------------------------------------------

    #initialization:
    #--------------------

    N = ones(1,M);

    # global couplings
    wgs=Gs;
    numWg=length(wgs);

    Coef=zeros(1,numWg);
    CoefU=zeros(1,numWg);
    CoefR=zeros(1,numWg);


    for j=1:numWg

        we=wgs(j);
        display(sprintf('we=#g',we))


        # initial condition:

        mu=0.001*rand(2*M,1);


        # Coupling matrix:
        #----------------------------------

        W11 = JN*we*C + w*JN*eye(M);
        W12 = diag(-Jo);
        W21 = JN*eye(M);
        W22 = -eye(M);
        Wmat = [W11 W12;W21 W22];



        for t=2:1:length(tspan)

            u = Wmat*mu + Io;
            re = feval(He,u(1:M));
            re = gamma*re./1000;
            ri = feval(Hi,u(M+1:end));
            ri = ri./1000;

            ke = -mu(1:M)/tau_exc+(1-mu(1:M)).*re;
            ki = -mu(M+1:end)/tau_inh + ri;

            kei = [ke;ki];

            mu=mu+dt*kei;

            mu(mu>1)=1;
            mu(mu<0)=0;

        end

        # Jacobian matrix:
        #--------------------

        u=Wmat*mu+Io;
        ue=u(1:M);
        ui=u(M+1:end);
        mue=mu(1:M);
        mui=mu(M+1:end);

        J11 =zeros(M);

        for p=1:M
            hoe=gamma/1000*feval(He,ue(p));
            h1e=gamma/1000*feval(H_1e,ue(p));
            for q=1:M
                if p==q
                    J=-1/tau_exc+(1-mu(p))*h1e*Wmat(p,p)-hoe;
                else
                    J=(1-mu(p))*h1e*Wmat(p,q);
                end
                J11(p,q)=J;
            end
        end


        hoe=gamma/1000*feval(He,ue);
        h1e=gamma/1000*feval(H_1e,ue);
        hoi=1/1000*feval(Hi,ui);
        h1i=1/1000*feval(H_1i,ui);

        J12 = diag(-Jo.*(1-mue).*h1e);

        J21 = diag(JN*h1i);

        J22 = diag(-1/tau_inh - h1i);

        Jmat = [J11 J12;J21 J22];



        # Eigen-decomposition of the Jacobian Matrix:

        [V D]=eig(Jmat);
        eigenval=diag(D);

        # Correlation matrix (linear approximation):
        #-------------------------------------------

        # get covariance matrix using the analytical relation:

        d=diag(D);
        P=zeros(2*M);
        U=V';
        q=(beta*dt)^2*eye(2*M);
        Q=V\q/U;
        for m=1:2*M
            for n=1:2*M
                P(m,n)=-Q(m,n)./(d(m)+d(n));
            end
        end
        rho=V*P*V';

        # correlation matrix (between exc. variables)
        Corr=zeros(M);
        for m=1:M
            for n=1:M
                Corr(m,n)=rho(m,n)/( sqrt(rho(m,m))*sqrt(rho(n,n)) );
            end
        end



        # linear approximation to get the covariance for the rate variables:
        #--------------------------------------------------------------------

        # covariance of the u-variables:

        Cv_u = Wmat*rho*Wmat';
        Cue = Cv_u(1:M,1:M);

        CorrU=zeros(M);
        for m=1:M
            for n=1:M
                CorrU(m,n)=Cue(m,n)/( sqrt(Cue(m,m))*sqrt(Cue(n,n)) );
            end
        end



        # A: jacobian of the transfer function evaluated at the steady state:

        A = feval(H_1e,ue);
        A = diag(A);

        Cv = A*Cue*A';

        CorrR=zeros(M);
        for m=1:M
            for n=1:M
                CorrR(m,n)=Cv(m,n)/( sqrt(Cv(m,m))*sqrt(Cv(n,n)) );
            end
        end

        # Fitting Corr vs. FC_emp
        fcm    = CorrR(Indices); # Vector containing all the FC values below the diagonal
        fcm_z=atanh(fcm);


        rc      = corrcoef(fcm_z,fc_z);
        Coef(j) = rc(1,2);

    end #----------------------------------------------------------------

    fitting=Coef;

    figure
    plot(wgs,fitting,'r')
