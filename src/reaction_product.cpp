#include "openmc/reaction_product.h"

#include <string> // for string

#include <fmt/core.h>

#include "openmc/endf.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/memory.h"
#include "openmc/particle.h"
#include "openmc/random_lcg.h"
#include "openmc/secondary_correlated.h"
#include "openmc/secondary_kalbach.h"
#include "openmc/secondary_nbody.h"
#include "openmc/secondary_uncorrelated.h"
#include "openmc/secondary_thermal.h"

namespace openmc {

//==============================================================================
// ReactionProduct implementation
//==============================================================================

ReactionProduct::ReactionProduct(hid_t group)
{
  // Read particle type
  std::string temp;
  read_attribute(group, "particle", temp);
  particle_ = str_to_particle_type(temp);

  // Read emission mode and decay rate
  read_attribute(group, "emission_mode", temp);
  if (temp == "prompt") {
    emission_mode_ = EmissionMode::prompt;
  } else if (temp == "delayed") {
    emission_mode_ = EmissionMode::delayed;
  } else if (temp == "total") {
    emission_mode_ = EmissionMode::total;
  }

  // Read decay rate for delayed emission
  if (emission_mode_ == EmissionMode::delayed) {
    if (attribute_exists(group, "decay_rate")) {
      read_attribute(group, "decay_rate", decay_rate_);
    } else if (particle_ == ParticleType::neutron) {
      warning(fmt::format("Decay rate doesn't exist for delayed neutron "
                          "emission ({}).",
        object_name(group)));
    }
  }

  // Read secondary particle yield
  yield_ = read_function(group, "yield");

  int n;
  read_attribute(group, "n_distribution", n);

  for (int i = 0; i < n; ++i) {
    std::string s {"distribution_"};
    s.append(std::to_string(i));
    hid_t dgroup = open_group(group, s.c_str());

    // Read applicability
    if (n > 1) {
      hid_t app = open_dataset(dgroup, "applicability");
      applicability_.emplace_back(app);
      close_dataset(app);
    }

    // Determine distribution type and read data
    read_attribute(dgroup, "type", temp);
    if (temp == "uncorrelated") {
      distribution_.push_back(make_unique<UncorrelatedAngleEnergy>(dgroup));
    } else if (temp == "correlated") {
      distribution_.push_back(make_unique<CorrelatedAngleEnergy>(dgroup));
    } else if (temp == "nbody") {
      distribution_.push_back(make_unique<NBodyPhaseSpace>(dgroup));
    } else if (temp == "kalbach-mann") {
      distribution_.push_back(make_unique<KalbachMann>(dgroup));
    }

    close_group(dgroup);
  }
}

void ReactionProduct::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  auto n = applicability_.size();
  if (n > 1) {
    double prob = 0.0;
    double c = prn(seed);
    for (int i = 0; i < n; ++i) {
      // Determine probability that i-th energy distribution is sampled
      prob += applicability_[i](E_in);

      // If i-th distribution is sampled, sample energy from the distribution
      if (c <= prob) {
        distribution_[i]->sample(E_in, E_out, mu, seed);
        break;
      }
    }
  } else {
    // If only one distribution is present, go ahead and sample it
    distribution_[0]->sample(E_in, E_out, mu, seed);
    
  }
}

void ReactionProduct::get_pdf(
  double det_pos[3],double E_in,double& E_out,double mymu, uint64_t* seed , Particle &p,std::vector<double> &pdfs_cm , std::vector<double> &pdfs_lab ,std::vector<Particle> &ghost_particles) const
{
  double mypdf = 0;

  int distribution_index;
  auto n = applicability_.size();
  if (n > 1) {
    double prob = 0.0;
    double c = prn(seed);
    for (int i = 0; i < n; ++i) {
      // Determine probability that i-th energy distribution is sampled
      prob += applicability_[i](E_in);

      // If i-th distribution is sampled, sample energy from the distribution
      if (c <= prob) {
        //distribution_[i]->sample(E_in, E_out, mu, seed);
        distribution_index = i;
        break;
      }
    }
  } else {
    // If only one distribution is present, go ahead and sample it
    //distribution_[0]->sample(E_in, E_out, mu, seed);
    distribution_index = 0;
    
  }
 // now extract pdf 

AngleEnergy* angleEnergyPtr = distribution_[distribution_index].get();

if (CorrelatedAngleEnergy* correlatedAE = dynamic_cast<CorrelatedAngleEnergy*>(angleEnergyPtr)) {
   // std::cout << "Used " << typeid(*correlatedAE).name() << " implementation." << std::endl;
    (*correlatedAE).get_pdf(det_pos,E_in,E_out,seed ,p,pdfs_cm ,pdfs_lab ,ghost_particles);
    // Handle CorrelatedAngleEnergy
} else if (KalbachMann* kalbachMann = dynamic_cast<KalbachMann*>(angleEnergyPtr)) {
    //std::cout << "Used " << typeid(*kalbachMann).name() << " implementation." << std::endl;
    (*kalbachMann).get_pdf(det_pos,E_in,E_out,seed ,p,pdfs_cm ,pdfs_lab ,ghost_particles);
   // std::cout << "mypdf " << (*kalbachMann).get_pdf(E_in,E_out, 0.3333 , seed) << std::endl;
   // std::cout << " my E_in " << E_in <<std::endl;
   // std::cout << " my E out " << E_out <<std::endl;
    // Handle KalbachMann
} else if (NBodyPhaseSpace* nBodyPS = dynamic_cast<NBodyPhaseSpace*>(angleEnergyPtr)) {
    //std::cout << "Used " << typeid(*nBodyPS).name() << " implementation." << std::endl;
    (*nBodyPS).get_pdf(det_pos,E_in,E_out,seed ,p,pdfs_cm ,pdfs_lab ,ghost_particles);
    // Handle NBodyPhaseSpace
} else if (UncorrelatedAngleEnergy* uncorrelatedAE = dynamic_cast<UncorrelatedAngleEnergy*>(angleEnergyPtr)) {
    //std::cout << "Used " << typeid(*uncorrelatedAE).name() << " implementation." << std::endl;
    (*uncorrelatedAE).get_pdf(det_pos,E_in,E_out,seed ,p,pdfs_cm ,pdfs_lab ,ghost_particles);
    // Handle UncorrelatedAngleEnergy
} else if (CoherentElasticAE* coherentElasticAE = dynamic_cast<CoherentElasticAE*>(angleEnergyPtr)) {
    std::cout << "Used " << typeid(*coherentElasticAE).name() << " implementation." << std::endl;
    // Handle CoherentElasticAE
} else if (IncoherentElasticAE* incoherentElasticAE = dynamic_cast<IncoherentElasticAE*>(angleEnergyPtr)) {
    std::cout << "Used " << typeid(*incoherentElasticAE).name() << " implementation." << std::endl;
    // Handle IncoherentElasticAE
} else if (IncoherentElasticAEDiscrete* incoherentElasticAEDiscrete = dynamic_cast<IncoherentElasticAEDiscrete*>(angleEnergyPtr)) {
    std::cout << "Used " << typeid(*incoherentElasticAEDiscrete).name() << " implementation." << std::endl;
    // Handle IncoherentElasticAEDiscrete
} else if (IncoherentInelasticAEDiscrete* incoherentInelasticAEDiscrete = dynamic_cast<IncoherentInelasticAEDiscrete*>(angleEnergyPtr)) {
    std::cout << "Used " << typeid(*incoherentInelasticAEDiscrete).name() << " implementation." << std::endl;
    // Handle IncoherentInelasticAEDiscrete
} else if (IncoherentInelasticAE* incoherentInelasticAE = dynamic_cast<IncoherentInelasticAE*>(angleEnergyPtr)) {
    std::cout << "Used " << typeid(*incoherentInelasticAE).name() << " implementation." << std::endl;
    // Handle IncoherentInelasticAE
} else if (MixedElasticAE* mixedElasticAE = dynamic_cast<MixedElasticAE*>(angleEnergyPtr)) {
    std::cout << "Used " << typeid(*mixedElasticAE).name() << " implementation." << std::endl;
    // Handle MixedElasticAE
} else {
    std::cout << "Unknown derived type." << std::endl;
}


 //return mypdf;
}

} // namespace openmc
