/**
 * Basic Wrapper for EFT Morphing code
 *
 *
 */

#include "RandomEFT/EFTCrossSection.h" 
#include "RandomEFT/EFTMorphing.h" 
#include "TH1F.h"
#include "TFile.h"
#include "TGraph.h"
#include <iostream> 
#include <string>
#include <sstream>
#include <iomanip>
 
using namespace std;

// Simple EF TMorphing wrapper code for coupling 
// and cross section computation
class MorphWrapper{
public:
  MorphWrapper(){ weights = new float[nsamples]; };
  ~MorphWrapper(){ delete weights; };
  void setSampleData(const char *in_data);
  float * getWeights();
  const char * getCrossSections();
  void printData();

private:
  int ncouplings;
  int nsamples;
  vector<float> morphed;
  vector<string> types;
  vector<vector<float> > prod;
  vector<vector<float> > dec;
  vector<vector<float> > same;
  vector<vector<float> > samples;
  float *weights;
};

void MorphWrapper::setSampleData(const char *in_data){
  // Resetting old data in case is being called again
  morphed.clear(); types.clear(); prod.clear(); dec.clear(); same.clear(); samples.clear(); 

  string data_string(in_data,strlen(in_data));
  //cout << in_data << endl;

  stringstream ss(data_string);
  // read n samles and ncouplings
  ss >> nsamples >> ncouplings;

  // read types from string for each coupling
  string type;
  for (int i = 0; i < ncouplings; i++){
    ss >> type;
    types.push_back(type);
  }

  // reading target sample
  float target;
  for (int i = 0; i < ncouplings; i++){
    ss >> target;
    morphed.push_back(target);
  }

  // reading each sample

  float sample_coupling;
  for (int i = 0; i < nsamples; i++){
    samples.push_back(vector<float>());
    for (int j = 0; j < ncouplings;j++){
      ss >> sample_coupling;
      samples[i].push_back(sample_coupling);
    }
  }
  //cout<<"Basis samples "<<endl;
  //calculating N eff for this basis.   
  // making vectors of couplings for this basis
  for(unsigned int i  = 0; i<nsamples; ++i)
  {
    vector<float> s = samples[i];

    vector<float>  prod_l;
    vector<float>  dec_l; 
    vector<float>  same_l; 

    //cout<<endl;

    //cout<<"{";
    for (unsigned int j  = 0; j<ncouplings; ++j)
    {
      //cout<< s[j];
      //if(j<(ncouplings-1)) cout<<"," << " ";
      if(types[j] == "P")      prod_l.push_back(s[j]);
      else if(types[j] == "D") dec_l.push_back(s[j]);
      else if(types[j] == "S") same_l.push_back(s[j]);
      else{
        cout<<"Unknown coupling type"<<endl;

      }//end of the type selection loop                                                                                                                                      
    }//end of sample couplings print loop                                                                                                                                   
    //cout<< "}," << endl;


    prod.push_back(prod_l);
    dec.push_back(dec_l); 
    same.push_back(same_l);

  }//end of loop over all the samples
}

float* MorphWrapper::getWeights(){
  //morphing class  
  EFTMorphing * mph  = new EFTMorphing(prod, dec, same);
  //vector of input couplings for testing point:
  //same size and same order as basis  
  //have to change this
  vector<float> test_p;
  vector<float> test_d;
  vector<float> test_s;
  

  for (unsigned int j  = 0; j<ncouplings; ++j)
  {
    if(types[j] == "P")      test_p.push_back(morphed[j]);
    else if(types[j] == "D") test_d.push_back(morphed[j]);
    else if(types[j] == "S") test_s.push_back(morphed[j]);
  }//end of sample couplings print loop                                                                                                                                   
  //cout << "MORPHING VALUES" << endl;
  vector<float> mc =  mph->morphingCoefficients(test_p, test_d, test_s);

  ostringstream out_data;

  for(unsigned int i =0;i<mc.size();++i) 
  { 
    //cout << fixed << setw( 11 ) << setprecision( 6 ) << mc[i] << endl;
    weights[i] = mc[i];
    out_data <<  fixed << setw( 11 ) << setprecision( 6 ) << mc[i] << ' ';
    //n_eff += mc[i] * cross_sections[i];
    //cs_norm += TMath::Abs(mc[i])* cross_sections[i];
  }
  
  //  cout<<endl;
  //  cout<<" Effective number of events: "<<n_eff<<endl;
  //  std::cout<<"full CS "<<cs_norm<<std::endl;

  //memory cleanup
  delete mph;
  mph = 0L;
  // TODO: This can bee dangereous, have to look for other options
  //return out_data.str().c_str() ;
  return weights;
}

const char* MorphWrapper::getCrossSections(){
  float kSM     = 0.;
  float kHZZ    = 0.;
  float kAZZ    = 0.;
  float kHWW    = 0.;
  float kAWW    = 0.;
  float kHdZ    = 0.;
  float kHdWR   = 0.;
  float kHdWI   = 0.;
  float kHgamgam= 0.;
  float kAgamgam= 0.;
  float kHZgam  = 0.;
  float kAZgam  = 0.;
  float kHdgam  = 0.;
  float kHgg	 = 0.;
  float kAgg    = 0.;
  float calpha  = 0.707;
  float dec_f   = 0.1151;
  ostringstream out_data;
  // just for energy 8.TeV
  EFTCrossSection * eft_cs = new EFTCrossSection(8.);

  vector<float> cross_section;
  float cs;
  for (int i = 0; i < nsamples; i++){
    //TODO: For now just working on VBF->4ll data
    kSM = samples[i][0];
    kHZZ = samples[i][1]*16.247;
    kAZZ = samples[i][2]*16.247;
    cs = eft_cs->vbfToDiFermions(kSM, kHZZ, kAZZ, kHWW, kAWW, kHdZ,  
                                      kHdWR, kHdWI, kHgamgam, kAgamgam, 
                                      kHZgam, kAZgam, kHdgam, kHgg, 
                                      kAgg, calpha,dec_f);
    //cout << cs << endl;
    out_data << cs << ' ';

  }
  delete eft_cs;
  return out_data.str().c_str();
}

extern "C"{
  MorphWrapper* MorphingWrapperNew(){ return new MorphWrapper(); }
  void setSampleData(MorphWrapper *mw,const char* in_data){return mw->setSampleData(in_data);}
  float* getWeights(MorphWrapper *mw){ return mw->getWeights();}
  const char* getCrossSections(MorphWrapper *mw){return mw->getCrossSections();}
}

