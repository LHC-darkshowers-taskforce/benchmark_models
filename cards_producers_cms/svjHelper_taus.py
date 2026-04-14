import os, math
from string import Template
from glob import glob
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class quark(object):
    def __init__(self,id,mass,charge):
        self.id = id
        self.mass = mass
        self.charge = charge
        self.massrun = mass
        self.color = 3
        self.bf = 1           #theory branching ratios
        self.bf_scaled = 1    #scaled branching ratios by rinv value
        self.on = True        #phase space allowed decay
        self.active = True    #for running nf

    def __repr__(self):
        return str(self.id)+": m = "+str(self.mass)+", mr = "+str(self.massrun)+", on = "+str(self.on)+", bf = "+str(self.bf)

#Added leptons class

class lepton(object):
    def __init__(self,id,mass,charge):
        self.id = id
        self.mass = mass
        self.charge = charge
        self.color = 1
        self.bf = 1           #theory branching ratios
        self.bf_scaled = 1    #scaled branching ratios by rinv value
        self.on = True        #phase space allowed decay  
  
    def __repr__(self):
        return str(self.id)+": m = "+str(self.mass)+", on = "+str(self.on) +", bf = "+str(self.bf)


# follows Ellis, Stirling, Webber calculations
class massRunner(object):
    def __init__(self):
        # QCD scale in GeV
        self.Lambda = 0.218

    # RG terms, assuming nc = 3 (QCD)
    def c(self): return 1./math.pi
    def cp(self,nf): return (303.-10.*nf)/(72.*math.pi)
    def b(self,nf): return (33.-2.*nf)/(12.*math.pi)
    def bp(self,nf): return (153.-19.*nf)/(2.*math.pi*(33.-2.*nf))
    def alphaS(self,Q,nf): return 1./(self.b(nf)*math.log(Q**2/self.Lambda**2))

    # derived terms
    def cb(self,nf): return 12./(33.-2.*nf)
    def one_c_cp_bp_b(self,nf): return 1.+self.cb(nf)*(self.cp(nf)-self.bp(nf))

    # constant of normalization
    def mhat(self,mq,nfq):
        return mq/math.pow(self.alphaS(mq,nfq),self.cb(nfq))/self.one_c_cp_bp_b(nfq)

    # mass formula
    def m(self,mq,nfq,Q,nf):
        # temporary hack: exclude quarks w/ mq < Lambda
        alphaq = self.alphaS(mq,nfq)
        if alphaq < 0: return 0
        else: return self.mhat(mq,nfq)*math.pow(self.alphaS(Q,nf),self.cb(nf))*self.one_c_cp_bp_b(nf)

    # operation
    def run(self,quark,nfq,scale,nf):
        # run to specified scale and nf
        return self.m(quark.mass,nfq,scale,nf)

class quarklist(object):
    def __init__(self):
        # mass-ordered
        self.qlist = [
            quark(2,0.0023,0.67), # up
            quark(1,0.0048,0.33), # down
            quark(3,0.095,0.33),  # strange
            quark(4,1.275,0.67),  # charm
            quark(5,4.18,0.33),   # bottom
        ]
        self.scale = None
        self.runner = massRunner()

    def set(self,scale):
        self.scale = scale
        # mask quarks above scale
        for q in self.qlist:
            # for decays
            if scale is None or 2*q.mass < scale: q.on = True
            else: q.on = False
            # for nf running
            if scale is None or q.mass < scale: q.active = True
            else: q.active = False
        # compute running masses
        if scale is not None:
            qtmp = self.get(active=True)
            nf = len(qtmp)
            for iq,q in enumerate(qtmp):
                q.massrun = self.runner.run(q,iq,scale,nf)
        # or undo running
        else:
            for q in self.qlist:
                q.massrun = q.mass

    def reset(self):
        self.set(None)

    def get(self,active=False):
        return [q for q in self.qlist if (q.active if active else q.on)]

#### Leptons list

class leptonslist(object):
    def __init__(self):
        # mass-ordered
        self.llist = [
            lepton(11,0.0005109989461,1),   # electrons
            lepton(13,0.1056583745,1),    # muons
            lepton(15,1.77686,1),        # taus
        ]
        self.scale = None    

    def set(self,scale):
        self.scale = scale
        # mask quarks above scale - phase space allowed decay                                                                                                                    
        for l in self.llist:
            # for decays                                                                                                                                                        
            if scale is None or 2.0*l.mass < scale: 
               l.on = True
            else: 
                l.on = False        

    def reset(self):
        self.set(None)

    def get(self,active=False):
        return [l for l in self.llist if l.on]


class svjHelper(object):
    def __init__(self):
        self.quarks_pseudo = quarklist()
        self.quarks_vector = quarklist()
        self.leptons_pseudo = leptonslist()
        self.leptons_vector = leptonslist()
        self.quarks = quarklist()
        self.alphaName = ""
        self.generate = None
        # parameters for lambda/alpha calculations
        self.n_c = 3
        self.n_f = 2
        self.b0 = 11.0/6.0*self.n_c - 2.0/6.0*self.n_f

    def setAlpha(self,alpha,svjl,lambdaHV):
        self.alphaName = alpha
        # "empirical" formula
        if not svjl:
            lambda_peak = 3.2*math.pow(self.mDark,0.8)
            if self.alphaName=="peak":
                self.alpha = self.calcAlpha(lambda_peak)
            elif self.alphaName=="high":
                self.alpha = 1.5*self.calcAlpha(lambda_peak)
            elif self.alphaName=="low":
                self.alpha = 0.5*self.calcAlpha(lambda_peak)
            else:
                raise ValueError("unknown alpha request: "+alpha)
        else:
             self.alpha = self.calcAlpha(lambdaHV)


    # Calculation of rho mass from Lattice QCD fits (arXiv:2203.09503v2)                                                                                                                                     
    def calcLatticePrediction(self,mPiOverLambda,mPseudo):
        mVectorOvermPseudo = (1.0/mPiOverLambda)*math.pow(5.76 + 1.5*math.pow(mPiOverLambda,2) ,0.5)
        mVector = mVectorOvermPseudo*mPseudo
        return mVector

    def calcAlpha(self,lambdaHV):
        return math.pi/(self.b0*math.log(1000/lambdaHV))

    def calcLambda(self,alpha):
        return 1000*math.exp(-math.pi/(self.b0*alpha))

    # has to be "lambdaHV" because "lambda" is a keyword
    def setModel(self,mMediator,rinv,mPiOverLambda,lambdaHV,BRtau, events, com):
        
        # store the basic parameters
        self.mMediator = mMediator
        
        if (mPiOverLambda <= 1.5):
            self.mPiOverLambda = mPiOverLambda
        else:
            print("mPiOverLambda must be smaller than 1.5 to get rho > pi pi decays!!")

        self.lambdaHV = lambdaHV
        self.mPseudo = self.mPiOverLambda * self.lambdaHV
        self.mVector = self.calcLatticePrediction(self.mPiOverLambda,self.mPseudo)
        self.BRtau = BRtau

        self.rinv = rinv

        # get more parameters
        self.mMin = self.mMediator-1
        self.mMax = self.mMediator+1
        
        self.mSqua = self.lambdaHV + 0.2 # dark scalar quark mass (also used for pTminFSR)

       
        self.quarks_pseudo.set(self.mPseudo)
        self.quarks_vector.set(self.mVector)
        self.leptons_pseudo.set(self.mPseudo)
        self.leptons_vector.set(self.mVector)
        self.alpha = self.calcAlpha(self.lambdaHV)

        self.events = events
        self.com = com
            
    

    def getOutName(self):
        _outname = "SVJTau"
        
        _outname += "_mZprime-{:g}".format(self.mMediator)
        _outname += "_rinv-{:g}".format(self.rinv)
        _outname += "_BRtau-{:g}".format(self.BRtau)
        _outname += "_mPioverLambda-{:g}".format(self.mPiOverLambda)
        _outname += "_lambdaHV-{:g}".format(self.lambdaHV)

            
        return _outname



    def invisibleDecay(self,mesonID,dmID):
        lines = ['{:d}:oneChannel = 1 {:g} 0 {:d} -{:d}'.format(mesonID,self.rinv,dmID,dmID)]
        return lines


    #Here needs to be modified in case of Tau models with Mass insertion for A'    
    def pseudo_scalar_visibleDecay(self,type,mesonID):
       
        theQuarks = self.quarks_pseudo.get()
        theLeptons = self.leptons_pseudo.get()
       

        if type=="Taus_effective":
            
            bfLeptons = (1.0-self.rinv)*self.BRtau
            bfQuarks = (1.0-self.rinv)*(1.0 - self.BRtau)
            print(bfLeptons)
            print(bfQuarks)
            for iq,q in enumerate(theQuarks):
                print("quarks idx",iq)
                if (iq == 3):
                    theQuarks[iq].bf = bfQuarks
                else:
                    theQuarks[iq].bf = 0.0

            for il,l in enumerate(theLeptons):
                print("leptons idx",il)
                if (il == 2):
                    theLeptons[il].bf = bfLeptons
                else:
                    theLeptons[il].bf = 0.0
      


        else:
            raise ValueError("unknown visible decay type: "+type)
            
        lines = []

        if (type=="Taus_effective"):
            # lines for decays to quarks                                                                                                                    
            lines_leptons = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(mesonID,q.bf,q.id,q.id) for q in theQuarks if q.bf>0]
            # lines for decays to leptons                                                                                                
            lines_quarks = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(mesonID,l.bf,l.id,l.id) for l in theLeptons if l.bf>0]
            lines = lines_leptons + lines_quarks

        return lines


    

### Internal decays rho to pi pi
    def vector_internalDecay(self,type,mesonID):
        br = 0 
        if type=="Internal_simple":
            if (mesonID == 4900113):
               br = 1.0  
               decay_prod_1 = 4900211
               decay_prod_2 = -4900211
            if (mesonID == 4900213):
               br = 1.0  
               decay_prod_1 = 4900111
               decay_prod_2 = 4900211
        # lines for decays to quarks                                                                                                                              
        lines_rho_to_pipi = [
             '{:d}:mayDecay=on'.format(mesonID),
             '{:d}:oneChannel = 1 {:g} 91 {:d} {:d}'.format(mesonID,br,decay_prod_1,decay_prod_2),
        ]
        
        # lines for decays to leptons                                                                                                                              
        return lines_rho_to_pipi


    def getPythiaSettings(self):
        
        #general settings
        lines_settings = [
        'Main:numberOfEvents = {}'.format(self.events) ,
        'Main:timesAllowErrors = 3',
        'Random:setSeed = on',
        'Random:seed = 0',
        'Init:showChangedSettings = on',
        'Init:showAllSettings = on',
        'Init:showChangedParticleData = on',
        'Init:showAllParticleData = on',
        'Next:numberCount = 1000',
        'Next:numberShowLHA = 1',
        'Next:numberShowInfo = 1',
        'Next:numberShowProcess = 1',
        'Next:numberShowEvent = 1',
        'Stat:showPartonLevel = on',
        '',
        '',
        ]
        
        #lines beams settings
        lines_beams_settings = [
            'Beams:idA = 2212',
            'Beams:idB = 2212',
            'Beams:eCM = {:g}'.format(self.com),
            '',
            '',
        ]

        lines_schan = [
            # parameters for leptophobic Z'
            '4900023:m0 = {:g}'.format(self.mMediator),
            '4900023:mMin = {:g}'.format(self.mMin),
            '4900023:mMax = {:g}'.format(self.mMax),
            '4900023:mWidth = 0.01',
            '4900023:oneChannel = 1 0.982 102 4900101 -4900101',
            # SM quark couplings needed to produce Zprime from pp initial state
            '4900023:addChannel = 1 0.003 102 1 -1',
            '4900023:addChannel = 1 0.003 102 2 -2',
            '4900023:addChannel = 1 0.003 102 3 -3',
            '4900023:addChannel = 1 0.003 102 4 -4',
            '4900023:addChannel = 1 0.003 102 5 -5',
            '4900023:addChannel = 1 0.003 102 6 -6',
            # decouple
            '4900001:m0 = 50000',
            '4900002:m0 = 50000',
            '4900003:m0 = 50000',
            '4900004:m0 = 50000',
            '4900005:m0 = 50000',
            '4900006:m0 = 50000',
            '4900011:m0 = 50000',
            '4900012:m0 = 50000',
            '4900013:m0 = 50000',
            '4900014:m0 = 50000',
            '4900015:m0 = 50000',
            '4900016:m0 = 50000',
            '',
            '',
        ]

    
        
        lines_schan.extend([
                'HiddenValley:ffbar2Zv = on',
        ])
            

        #Setting hidden spectrum    
        # hidden spectrum:                                                                                                                                                                                        
        # fermionic dark quark,                                                                                                                                                                                           
        # diagonal pseudoscalar meson, off-diagonal pseudoscalar meson, DM stand-in particle,                                                                                                                             
        # diagonal vector meson, off-diagonal vector meson, DM stand-in particle
        
    
        lines_decay = [
                 '4900101:m0 = {:g}'.format(self.mSqua),
                 '4900111:m0 = {:g}'.format(self.mPseudo),
                 '4900211:m0 = {:g}'.format(self.mPseudo),
                 '51:m0 = 0.0',
                 '51:isResonance = false',
                 '4900113:m0 = {:g}'.format(self.mVector),
                 '4900213:m0 = {:g}'.format(self.mVector),
                 '53:m0 = 0.0',
                 '53:isResonance = false',
        ]    


        # other HV params
        lines_decay.extend([
            'HiddenValley:Ngauge = {:d}'.format(self.n_c),
            # when Fv has spin 0, qv spin fixed at 1/2
            'HiddenValley:spinFv = 0',
            'HiddenValley:FSR = on',
            'HiddenValley:fragment = on',
            'HiddenValley:alphaOrder = 1',
            'HiddenValley:setLambda = on',
            'HiddenValley:Lambda = {:g}'.format(self.lambdaHV),
            'HiddenValley:nFlav = {:d}'.format(self.n_f),
            'HiddenValley:probVector = 0.75',
        ])



        # branching - effective rinv (applies to all meson species b/c n_f >= 2)
        # pseudoscalars have mass insertion decay, vectors have democratic decay  
        lines_decay += self.invisibleDecay(4900111,51)
        lines_decay += self.invisibleDecay(4900211,51)
        lines_decay += self.pseudo_scalar_visibleDecay("Taus_effective",4900111)
        lines_decay += self.pseudo_scalar_visibleDecay("Taus_effective",4900211)
        lines_decay += self.vector_internalDecay("Internal_simple",4900113)
        lines_decay += self.vector_internalDecay("Internal_simple",4900213)
            

        return lines_settings + lines_beams_settings + lines_schan + lines_decay



if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mZprime", dest="mZprime", required=True, type=float, help="Zprime mass (GeV)")
    parser.add_argument("--rinv", dest="rinv", required=True, type=float, help="invisible fraction")
    parser.add_argument("--brtau", dest="brtau",  required=True, type=float, help="Branching fraction to tau leptons of dark pions")
    parser.add_argument("--mPiOverLambda", dest="mPiOverLambda", required=True, type=float, help="Lightest dark pseudoscalar mass over LambdaHV")
    parser.add_argument("--lambda", dest="lambdaHV", required=True, type=float, help="dark sector confinement scale")
    parser.add_argument("--events", dest="events", default=10000, type=int, help="number of events to generate")
    parser.add_argument("--com", dest="com", default=13000, type=float, help="center-of-mass energy (GeV)")
    args = parser.parse_args()


    #def setModel(self,mMediator,rinv,mPiOverLambda,lambdaHV,BRtau):

    helper = svjHelper()
    helper.setModel(args.mZprime, args.rinv, args.mPiOverLambda, args.lambdaHV, args.brtau,args.events, args.com)

    lines = helper.getPythiaSettings()
    fname = helper.getOutName()+".txt"
    with open(fname,'w') as file:
        file.write('\n'.join(lines))

