# --------------------------------------------------------------------------------------
# Full pipeline for ignitiion computation
# By Gustavo Deco
# Adapted to python by Gustavo Patow
# --------------------------------------------------------------------------------------

import numpy as np
import scipy.io as sio

def ignition(GrCV):
    tcrange = np.union1d(np.arange(0,34), np.arange(41,75))  # [1:34 42:75]
    C = GrCV[:, tcrange][tcrange, ]
    C=C/np.max(C)*0.2
    print(f'C shape is {C.shape}')
    N = 68

    # NSUB = 389
    # Ntrials = 10
    # Tmax = 616
    # indexsub = np.arange(0,NSUB)

    # ===========================================================================
    # Parameters for the mean field model
    # dtt   = 1e-3;   % Sampling rate of simulated neuronal activity (seconds)
    dt = 0.1  # as in the simulate_SimAndBOLD module definition...

    # taon=100  # as in the DMF model definition
    # taog=10   #           ✓
    # gamma=0.641  #        ✓
    # sigma=0.01  # as in the Euler-Maruyama integrator
    # JN=0.15  # as in the DMF model definition
    # I0=0.382  #           ✓
    # Jexte=1.  #           ✓
    # Jexti=0.7  #          ✓
    # w=1.4  #              ✓

    # #%%%%%%%%%%%
    # Optimize
    #
    # WE=0:0.01:3;
    #
    # we=WE(s);
    #
    # Jbal=Balance_J(we,C);
    #
    # J=0;
    #
    # #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # #% Check supraliminar Ignition (a la Dehaene)
    #
    SEED = np.array([4, 10, 12, 20])
    PERTURB = np.arange(0., 0.2, 0.001)
    sigfunc = lambda A,x : A(0) / (1 + np.exp(-A(1)*(x-A(2)))) + A(3)  # sigfunc = @(A, x)(A(1) ./ (1 + exp(-A(2)*(x-A(3)))) + A(4))
    # options = optimset('MaxFunEvals',10000,'MaxIter',1000)
    #
    # clear Ignition3 Igthreshold33 Igthreshold3;
    # nseed=1;
    # Tmax=7000;
    # SUBTR=10;
    # nump=length(PERTURB);
    # neuro_act2=zeros(SUBTR,Tmax+1,N);
    #
    # for seed=SEED   # check all single area stimulation
    #     fprintf('seed %f \n', seed);
    #     clear peakrate1 peak1 peakrate3 peakrate2 basal tdecay ignition1 ignitionthreshold1;
    #     tseed=1:N;
    #     tseed(find(seed==tseed))=[];
    #     tseed(find(N/2+seed==tseed))=[];
    #     kk=1;
    #     for perturb=PERTURB  %% for each stimulation..different strengths (Istim)
    #         Istim=zeros(N,1);
    #         for sub=1:SUBTR
    #             nn=1;
    #             sn=0.001*ones(N,1);
    #             sg=0.001*ones(N,1);
    #             for t=0:dt:Tmax
    #                 if t==3000
    #                     Istim(seed)=perturb;
    #                     Istim(N/2+seed)=perturb;
    #                 elseif t==3500
    #                     Istim(seed)=0;
    #                     Istim(N/2+seed)=0;
    #                 end
    #                 xn=Istim+I0*Jexte+JN*w*sn+we*JN*C*sn-Jbal.*sg-J.*sg;
    #                 xg=I0*Jexti+JN*sn-sg;
    #                 rn=phie(xn);
    #                 rg=phii(xg);
    #                 sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+sqrt(dt)*sigma*randn(N,1);
    #                 sn(sn>1) = 1;
    #                 sn(sn<0) = 0;
    #                 sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma*randn(N,1);
    #                 sg(sg>1) = 1;
    #                 sg(sg<0) = 0;
    #                 if abs(mod(t,1))<0.01
    #                     neuro_act2(sub,nn,:)=rn';
    #                     nn=nn+1;
    #                 end
    #             end
    #         end
    #         neuro_act1=squeeze(mean(neuro_act2(1:SUBTR,:,:),1));
    #         ntwi=1;
    #         for twi=1:20:nn-1-19
    #             neuro_actf(ntwi,:)=mean(neuro_act1(twi:twi+19,:));
    #             ntwi=ntwi+1;
    #         end
    #         ssnum=1;
    #         for ss=tseed
    #             peakrate3(ssnum)=max(neuro_actf(150:end,ss),[],1);
    #             basal(ssnum)=mean(neuro_actf(100:150,ss));
    #             sbasal=std(neuro_actf(100:150,ss),[],1);
    #             decayneuro=squeeze(neuro_actf(175:end,ss));
    #             tscale=0:1/length(decayneuro):1;
    #             bdecay=polyfit(tscale(1:length(decayneuro)),log(decayneuro'),1);
    #             tdecay(ssnum)=bdecay(1);
    #             ssnum=ssnum+1;
    #         end
    #         peakrate2(kk,:)=peakrate3./basal;
    #         kk=kk+1;
    #     end
    #     peakrate1=max(peakrate2,[],1);
    #     for ntarget=1:length(peakrate1)
    #       A0=[mean(peakrate2(end-10:end,ntarget))-mean(peakrate2(1:10,ntarget)) 10 0.1 mean(peakrate2(1:10,ntarget))];
    #       Afit = lsqcurvefit(sigfunc,A0,PERTURB,(peakrate2(:,ntarget))',[0 0 -1 0],[50 100 1 2*mean(peakrate2(1:10,ntarget))],options);
    #       yfit=Afit(1) ./ (1 + exp(-Afit(2)*(PERTURB-Afit(3))))+Afit(4);
    #       ignition1(ntarget)=max(diff(diff(yfit/0.001)/0.001));
    #       peak1(ntarget)=peakrate1(ntarget);
    #       [aux indigth]=max(diff(yfit/0.001));
    #       ignitionthreshold1(ntarget)=indigth;
    #     end
    #     Ignition2(nseed)=mean(ignition1);
    #     Ignitionthreshold2(nseed)=mean(ignitionthreshold1);
    #     Peak2(nseed)=mean(peak1);
    #     Decay2(nseed)=std(tdecay);
    #     nseed=nseed+1;
    # end
    # Ignition=mean(Ignition2);
    # Ignitionthreshold=mean(Ignitionthreshold2);
    # Maxrate=mean(Peak2);
    # Decay=mean(Decay2);
    #
    # fprintf('saving files...');
    # save(sprintf('WG_%03d.mat',s),'Ignition','Ignitionthreshold','Maxrate','Decay');

if __name__ == '__main__':
    s=1  # s=str2num(getenv('SLURM_ARRAY_TASK_ID'))  % for debug purposes only...

    dataPath = "../../Data_Raw/DecoEtAlDRAFT/"
    dataFile = 'SC_GenCog_PROB_30.mat'
    M = sio.loadmat(dataPath + dataFile); print('{} File contents:'.format(dataPath + dataFile), [k for k in M.keys()])
    GrCV = M['GrCV']

    ignition(GrCV)
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
