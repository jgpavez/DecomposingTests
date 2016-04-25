#include <TRandom1.h>
#include <TMath.h>
#include <TH1D.h>
#include <vector>
#include <TLorentzVector.h>

using namespace std;

void processNewData(const char* filename){

    TFile* file = new TFile(filename,"read");
    TNtupleD* tuple = file->Get("HiggsTree");

    //TObjArray *array = tuple->GetListOfBranches()->Clone();
    //cout << array->GetEntries() << endl;
    //for(int i = 0; i < array->GetEntries(); ++i) { cout << '"' << array->At(i)->GetName() << '"' << ','; }

    const int n_features = 51;
    //string features[n_features] = {"mH", "Z1_m", "Z2_m", "Mjj", "DelEta_jj", "DelPhi_jj", "jet1_eta", "jet2_eta", 
    //      "jet1_pt", "jet2_pt", "ZeppetaZZ", "pT_Hjj", "pT_Hjj_bin_50"}; // missing BSTscore, sf_weight, total_weight
    //string features[n_features] = {"Z1_E","Z1_pt","Z1_eta","Z1_phi","Z1_m","Z2_E","Z2_pt","Z2_eta","Z2_phi","Z2_m","higgs_E","higgs_pt","higgs_eta","higgs_phi","higgs_m","lep_E","lep_eta","lep_phi","lep_pt","lep_charge","lep_RecoID","lep_PID","jet_E","jet_eta","jet_phi","jet_pt","jet_m","jet_isPU","DelPhi_Hjj","mH","pT_Hjj","DelEta_jj","EtaProd_jj","DelY_jj","DelPhi_jj","DelR_jj","Mjj","Mjets","njets","jet1_E","jet1_eta","jet1_y","jet1_phi","jet1_pt","jet1_m","jet1_isPU","jet2_E","jet2_phi","jet2_eta","jet2_y","jet2_pt","jet2_m","jet2_isPU","DelPt_jj","minDelR_jZ","DelPt_ZZ","Zeppetaj3","ZeppetaZZ","jet3_E","jet3_eta","jet3_phi","jet3_pt","jet3_m","jet3_isPU"};
    string features[n_features] = {"Z1_E","Z1_pt","Z1_eta","Z1_phi","Z1_m","Z2_E","Z2_pt","Z2_eta","Z2_phi","Z2_m","higgs_E","higgs_pt","higgs_eta","higgs_phi","higgs_m","DelPhi_Hjj","mH","pT_Hjj","DelEta_jj","EtaProd_jj","DelY_jj","DelPhi_jj","DelR_jj","Mjj","Mjets","njets","jet1_E","jet1_eta","jet1_y","jet1_phi","jet1_pt","jet1_m","jet1_isPU","jet2_E","jet2_phi","jet2_eta","jet2_y","jet2_pt","jet2_m","jet2_isPU","DelPt_jj","minDelR_jZ","DelPt_ZZ","Zeppetaj3","ZeppetaZZ","jet3_E","jet3_eta","jet3_phi","jet3_pt","jet3_m","jet3_isPU"};
    const int new_n_features = 13;
    string new_features[new_n_features] = {"minDelR_jZ", "DelPhi_Hjj", "DelEta_Hjj", "DelPhi_jj", "DelR_jj", "DelY_jj", "Mjj", "DelPt_jj", "DelPt_ZZ", "pT_Hjj", "Mjets", "Zeppetaj3", "ZeppetaZZ"}
    vector <TLorentzVector> *lepton = 0;
    vector <TLorentzVector> *jet = 0;
    vector <TLorentzVector> *Z = 0;
    TLorentzVector *H;
    TLorentzVector *dijet;


    Int_t jet_n;
    Double_t values[new_n_features]; 
   
    ofstream writefile;
    writefile.open(Form("data_%s.dat", filename));

    int i,j;

    for( j = 0; j < new_n_features; j++){
        tuple->SetBranchAddress(new_features[j].c_str(), &values[j]);
    }
    

    tuple->SetBranchAddress("lepton_4v",&lepton);
    tuple->SetBranchAddress("jet_4v",&jet);
    tuple->SetBranchAddress("Zbosons_4v",&Z);
    tuple->SetBranchAddress("Hbosons_4v",&H);
    tuple->SetBranchAddress("dijet_4v",&dijet);

    tuple->SetBranchAddress("jet_n",&jet_n);


    Int_t entries = tuple->GetEntries();
    cout<<"Entries: "<<entries<<endl;

    for(i=0;i<entries;i++){
        tuple->GetEntry(i);
        for( j = 0; j < new_n_features; j++){
          if (j != 0)
            writefile << " ";
          writefile << values[j];
        }
        writefile << " ";
        writefile << jet_n << " ";

        writefile << Z->at(0).E() << " ";
        writefile << Z->at(0).Pt() << " ";
        writefile << Z->at(0).Eta() << " ";
        writefile << Z->at(0).Phi() << " ";
        writefile << Z->at(0).M() << " ";

        writefile << Z->at(1).E() << " ";
        writefile << Z->at(1).Pt() << " ";
        writefile << Z->at(1).Eta() << " ";
        writefile << Z->at(1).Phi() << " ";
        writefile << Z->at(1).M() << " ";

        writefile << H->E() << " ";
        writefile << H->Pt() << " ";
        writefile << H->Eta() << " ";
        writefile << H->Phi() << " ";
        writefile << H->M() << " ";

        for(int k = 0; k < jet_n; k++){
          if (k > 2)
            break;
          writefile << jet->at(k).E() << " ";
          writefile << jet->at(k).Eta() << " ";
          writefile << jet->at(k).Y() << " ";
          writefile << jet->at(k).Phi() << " ";
          writefile << jet->at(k).Pt() << " ";
          writefile << jet->at(k).M() << " ";
        }
        for(int k = 0; k < 3 - jet_n; k++){
          writefile << -999. << " ";
          writefile << -999. << " ";
          writefile << -999. << " ";
          writefile << -999. << " ";
          writefile << -999. << " ";
          writefile << -999. << " ";
         
        }

        writefile << endl;
    }   

    writefile.close();
    delete tuple;
    delete file;
}
