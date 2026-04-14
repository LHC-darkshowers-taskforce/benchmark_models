import os, math, sys, shutil
from string import Template
from glob import glob
import numpy as np
from datetime import date
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
    def __init__(self,card_author,events,com):
       
        self.quarks_pseudo = quarklist()
        self.quarks_vector = quarklist()
        self.leptons_pseudo = leptonslist()
        self.leptons_vector = leptonslist()
        self.quarks = quarklist()

        # parameters of Hidden sector
        self.n_c = 3
        self.n_f = 2
        
        #card author
        self.username = card_author
        
        #n events to generate
        self.events = events
        self.com = com


    # Calculation of rho mass from Lattice QCD fits (arXiv:2203.09503v2)
    def calcLatticePrediction(self,mPiOverLambda,mPseudo):        
        mVectorOvermPseudo = (1.0/mPiOverLambda)*math.pow(5.76 + 1.5*math.pow(mPiOverLambda,2) ,0.5) 
        mVector = mVectorOvermPseudo*mPseudo
        return mVector


    # has to be "lambdaHV" because "lambda" is a keyword
    def setModel(self,svjl_type,mMediator,rinv,mPiOverLambda,lambdaHV):

        #if self.mPiOverLambda < 1.5: raise ValueError("mPiOverLambda must be > 1.5 (no internal decays)")
        if mPiOverLambda < 1.5: raise ValueError("mPiOverLambda must be > 1.5 (no internal decays). Please reset mPiOverLambda value.")

        #define model type
        self.svjl_type = svjl_type
        # store the basic parameters
        self.mMediator = mMediator
        self.mPiOverLambda = mPiOverLambda
        self.lambdaHV = lambdaHV
        self.mPseudo = self.mPiOverLambda * self.lambdaHV
        self.mVector = self.calcLatticePrediction(self.mPiOverLambda,self.mPseudo)
        self.rinv = rinv
    
        # get more parameters
        self.mMin = self.mMediator-1
        self.mMax = self.mMediator+1
        self.mSqua = self.lambdaHV + 0.2 # dark quark constituent mass

        # get limited set of quarks for decays (check mDark against quark masses, compute running)
        self.quarks_pseudo.set(self.mPseudo)
        self.quarks_vector.set(self.mVector)
        self.leptons_pseudo.set(self.mPseudo)
        self.leptons_vector.set(self.mVector)

            
                
    #naming of the output datacard
    def getOutName(self):

        _outname = "SVJL"
        _outname += "-{}".format(self.svjl_type)
        _outname += "_mZprime-{:g}".format(self.mMediator)
        _outname += "_rinv-{:g}".format(self.rinv)
        _outname += "_mPioverLambda-{:g}".format(self.mPiOverLambda)
        _outname += "_lambdaHV-{:g}".format(self.lambdaHV)
            
        return _outname

    #implementation of rinv
    def invisibleDecay(self,mesonID,dmID):
        lines = ['{:d}:oneChannel = 1 {:g} 0 {:d} -{:d}'.format(mesonID,self.rinv,dmID,dmID)]
        return lines

    #Here mass insertion or democratic decay tables for pseudo-scalars Z' mediated decays    
    def pseudo_scalar_visibleDecay(self,type,mesonID,dmID):
        
        theQuarks = self.quarks_pseudo.get()

        if type=="simple":
            # just pick down quarks
            theQuarks = [q for q in theQuarks if q.id==1]
            theQuarks[0].bf = (1.0-self.rinv)
        elif type=="democratic":
            bfQuarks = (1.0-self.rinv)/float(len(theQuarks))
            for iq,q in enumerate(theQuarks):
                theQuarks[iq].bf = bfQuarks
        elif type=="massInsertion":
            denom = sum([q.massrun**2 for q in theQuarks])
            # hack for really low masses
            if denom==0.: return self.visibleDecay("democratic",mesonID,dmID)
            for q in theQuarks:
                q.bf = (1.0-self.rinv)*(q.massrun**2)/denom
        else:
            raise ValueError("unknown visible decay type: "+type)
            
            lines = []

        if (type == "simple" or type=="democratic" or type=="massInsertion"):
            lines = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(mesonID,q.bf,q.id,q.id) for q in theQuarks if q.bf>0]

        return lines


    def calcTotalWidth_A_prime_simplified(self,theLeptons,theQuarks):
        
        # 3 takes into account the color factor
        total_width = 0
        for iq,q in enumerate(theQuarks):
              total_width  = total_width + q.color*q.charge**2*self.mVector*np.sqrt(1.0-4.0*q.mass**2/self.mVector**2)*(1+2*q.mass**2/self.mVector**2)

        for il,l in enumerate(theLeptons):
              total_width  = total_width + l.color*l.charge**2*self.mVector*np.sqrt(1.0-4.0*l.mass**2/self.mVector**2)*(1+2*l.mass**2/self.mVector**2)
              
        return total_width     

    def decay_with_A_prime_partial_width_simplified(self,sm_particle,total_width):
         
         sm_particle.bf = sm_particle.color*sm_particle.charge**2*self.mVector*np.sqrt(1.0-4.0*sm_particle.mass**2/self.mVector**2)*(1+2*sm_particle.mass**2/self.mVector**2) / total_width    


    #democratic decays into all leptons generations
    def scale_branchings2rinv(self,theQuarks,theLeptons):

        #fixing proportions
        x_rho = theQuarks[1].bf/theQuarks[0].bf
        y_rho = theQuarks[1].bf/theLeptons[0].bf
        z_rho = theQuarks[4].bf/theLeptons[0].bf
 
        #define rescaled branchings (wash out mass effects - simplifed/to improve)
        Br_l_rho = (1.0 - self.rinv)/(3.0 + 2.0*(y_rho/x_rho) + 2.0*y_rho + z_rho)
        Br_d_rho = Br_l_rho*y_rho
        Br_u_rho = Br_d_rho/x_rho
        Br_b_rho = z_rho*Br_l_rho

        #fixing scaled branchings for quarks and leptons
        theLeptons[0].bf_scaled = Br_l_rho
        theLeptons[1].bf_scaled = Br_l_rho
        theLeptons[2].bf_scaled = Br_l_rho

        theQuarks[0].bf_scaled = Br_u_rho
        theQuarks[1].bf_scaled = Br_d_rho
        theQuarks[2].bf_scaled = Br_u_rho
        theQuarks[3].bf_scaled = Br_d_rho
        theQuarks[4].bf_scaled = Br_b_rho


    #vector meson visible decay from A'
    def vector_visibleDecay(self,type,mesonID,dmID):
        
        theQuarks = self.quarks_vector.get()
        theLeptons = self.leptons_vector.get()

        if type=="A-Democratic":
            #Calculate total width
            total_width = self.calcTotalWidth_A_prime_simplified(theLeptons,theQuarks)

            #decays to quarks fix branchings
            for iq,q in enumerate(theQuarks):
                self.decay_with_A_prime_partial_width_simplified(q,total_width)

            #decays to leptons fix branchings    
            for il,l in enumerate(theLeptons):
                self.decay_with_A_prime_partial_width_simplified(l, total_width)

            #rescale branchings with rinv    
            self.scale_branchings2rinv(theQuarks,theLeptons)

        # lines for decays to quarks
        lines_leptons = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(mesonID,q.bf_scaled,q.id,q.id) for q in theQuarks if q.bf>0]
        # lines for decays to leptons
        lines_quarks = ['{:d}:addChannel = 1 {:g} 91 {:d} -{:d}'.format(mesonID,l.bf_scaled,l.id,l.id) for l in theLeptons if l.bf>0]

        return lines_leptons + lines_quarks


    def getPythiaSettings(self):

        today = date.today()

        #Intro lines
        lines_intro = [
        """##############################################################################################""",
        """# Pythia 8 card for SVJL model, as defined in arXiv 2206.03909""",
        f"""# Z mass scale: {self.mMediator} GeV""",
        f"""# rinv: {self.rinv}""",
        f"""# Number colors: {self.n_c}""",
        f"""# Number flavors: {self.n_f}""",
        f"""# Scalar meson mass: {self.mPseudo} GeV""",
        f"""# Vector meson mass: {self.mVector} GeV""",
        f"""# Confinement scale: {self.lambdaHV} GeV""",
        f"""# Generated by {' '.join(self.username)} on {today.strftime("%B %d, %Y")}""",
        """# lines starting with ! or # are commented out""",
        """##############################################################################################""",
        '',
        '',
        ]


        #general settings
        lines_settings = [
        '! 1) Settings used in the main program.',
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
            '! 2) Beam parameter settings. Values below agree with default ones.',
            'Beams:idA = 2212',
            'Beams:idB = 2212',
            'Beams:eCM = {:g}'.format(self.com),
            '',
            '',
        ]

        #Lines s-channel
        lines_schan = [
            '! 3) Turn on the production process.',
            #HV process
            'HiddenValley:ffbar2Zv = on',
            # parameters for leptophobic Z'
            '4900023:m0 = {:g}'.format(self.mMediator),
            '4900023:mMin = {:g}'.format(self.mMin),
            '4900023:mMax = {:g}'.format(self.mMax),
            '4900023:mWidth = 0.01',
            '4900023:doForceWidth = on',
            '',
            '4900023:oneChannel = 1 0.982 102 4900101 -4900101',
            # SM quark couplings needed to produce Zprime from pp initial state
            '4900023:addChannel = 1 0.003 102 1 -1',
            '4900023:addChannel = 1 0.003 102 2 -2',
            '4900023:addChannel = 1 0.003 102 3 -3',
            '4900023:addChannel = 1 0.003 102 4 -4',
            '4900023:addChannel = 1 0.003 102 5 -5',
            '4900023:addChannel = 1 0.003 102 6 -6',
            '',
            '! 4) For MC efficiency let zprime decay only to dark quarks.',
            '4900023:onMode = off',
            '4900023:onIfAny = 4900101',
            '',
            '',
        ]


        #Setting hidden spectrum
        # hidden spectrum:
        # fermionic dark quark,
        # diagonal pseudoscalar meson, off-diagonal pseudoscalar meson, DM stand-in particle,
        # diagonal vector meson, off-diagonal vector meson, DM stand-in particle


        # HV params
        lines_hv = [
            '! 5) HiddenValley Settings',
            'HiddenValley:Ngauge = {:d}'.format(self.n_c),
            'HiddenValley:nFlav = {:d}'.format(self.n_f),
            # when Fv has spin 0, qv spin fixed at 1/2
            'HiddenValley:spinFv = 0',
            'HiddenValley:FSR = on',
            'HiddenValley:fragment = on',
            'HiddenValley:alphaOrder = 1',
            'HiddenValley:setLambda = on',
            'HiddenValley:Lambda = {:g}'.format(self.lambdaHV),
            'HiddenValley:pTminFSR = {:g}'.format(1.1*self.lambdaHV),
            'HiddenValley:probVector = 0.75',
            '',
            '',
        ]
        
        lines_hv.extend([
                '! 6) Dark sector mass spectrum',
                '4900101:m0 = {:g}'.format(self.mSqua),
                '4900111:m0 = {:g}'.format(self.mPseudo),
                '4900211:m0 = {:g}'.format(self.mPseudo),
                '51:m0 = 0.0',
                '51:isResonance = false',
                '4900113:m0 = {:g}'.format(self.mVector),
                '4900213:m0 = {:g}'.format(self.mVector),
                '53:m0 = 0.0',
                '53:isResonance = false',
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
                '! 7) HV decays',
        ])


        # branching - effective rinv (applies to all meson species)
        # pseudoscalars have mass insertion decays (from Z'), vectors have democratic decays (from A')
      
        #all pseudoscalars decays
        lines_hv += self.invisibleDecay(4900111,51)
        lines_hv += self.pseudo_scalar_visibleDecay("massInsertion",4900111,51)
        lines_hv += self.invisibleDecay(4900211,51)
        lines_hv += self.pseudo_scalar_visibleDecay("massInsertion",4900211,51)
        
        
        #all vector mesons decays
        lines_hv += self.invisibleDecay(4900113,53)
        lines_hv += self.vector_visibleDecay(self.svjl_type,4900113,53)
        lines_hv += self.invisibleDecay(4900213,53)
        lines_hv += self.vector_visibleDecay(self.svjl_type,4900213,53)
            
        lines = lines_intro + lines_settings + lines_beams_settings + lines_schan + lines_hv

        return lines


    
if __name__=="__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mZprime", dest="mZprime",  required=True, type=float, help="Zprime mass (GeV)")
    parser.add_argument("--rinv", dest="rinv", required=True, type=float, help="invisible fraction")
    parser.add_argument("--svjl_type", dest="svjl_type", required=False, default="A-Democratic", type=str, help="Versions of SVJL model (e.g. A-Democratic)")
    parser.add_argument("--mPiOverLambda", dest="mPiOverLambda", required=True, type=float, help="Lightest dark pseudoscalar mass over LambdaHV")
    parser.add_argument("--lambda", dest="lambdaHV", required=True, type=float, help="dark sector confinement scale")
    parser.add_argument("--events", dest="events", default=10000, type=int, help="number of events to generate")
    parser.add_argument("--com", dest="com", default=13000, type=float, help="center-of-mass energy (GeV)")
    parser.add_argument("--card_author", dest="card_author", required=False, type=str,nargs='+', default="", help="author of the generated card")
    args = parser.parse_args()


    helper = svjHelper(args.card_author, args.events,  args.com)
    helper.setModel(args.svjl_type,args.mZprime, args.rinv, args.mPiOverLambda, args.lambdaHV)

    lines = helper.getPythiaSettings()
    fname = helper.getOutName()+".cmnd"
    with open(fname,'w') as file:
        file.write('\n'.join(lines))
        
    print("Generated card and saved in file: ", fname)
    
    
        
        
        
