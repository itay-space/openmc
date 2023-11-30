#include "openmc/secondary_uncorrelated.h"

#include <string> // for string

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/random_dist.h"
#include "openmc/particle.h"
#include "openmc/nuclide.h"

namespace openmc {

//==============================================================================
// UncorrelatedAngleEnergy implementation
//==============================================================================

UncorrelatedAngleEnergy::UncorrelatedAngleEnergy(hid_t group)
{
  // Check if angle group is present & read
  if (object_exists(group, "angle")) {
    hid_t angle_group = open_group(group, "angle");
    angle_ = AngleDistribution {angle_group};
    close_group(angle_group);
  }

  // Check if energy group is present & read
  if (object_exists(group, "energy")) {
    hid_t energy_group = open_group(group, "energy");

    std::string type;
    read_attribute(energy_group, "type", type);
    using UPtrEDist = unique_ptr<EnergyDistribution>;
    if (type == "discrete_photon") {
      energy_ = UPtrEDist {new DiscretePhoton {energy_group}};
    } else if (type == "level") {
      energy_ = UPtrEDist {new LevelInelastic {energy_group}};
    } else if (type == "continuous") {
      energy_ = UPtrEDist {new ContinuousTabular {energy_group}};
    } else if (type == "maxwell") {
      energy_ = UPtrEDist {new MaxwellEnergy {energy_group}};
    } else if (type == "evaporation") {
      energy_ = UPtrEDist {new Evaporation {energy_group}};
    } else if (type == "watt") {
      energy_ = UPtrEDist {new WattEnergy {energy_group}};
    } else {
      warning(
        fmt::format("Energy distribution type '{}' not implemented.", type));
    }
    close_group(energy_group);
  }
}

void UncorrelatedAngleEnergy::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  // Sample cosine of scattering angle
  if (!angle_.empty()) {
    mu = angle_.sample(E_in, seed);
  } else {
    // no angle distribution given => assume isotropic for all energies
    mu = uniform_distribution(-1., 1., seed);
  }

  // Sample outgoing energy
  E_out = energy_->sample(E_in, seed);
}

void UncorrelatedAngleEnergy::get_pdf(
  double det_pos[3],double E_in,double& E_out,double mymu, uint64_t* seed , Particle &p,std::vector<double> &pdfs_cm , std::vector<double> &pdfs_lab ,std::vector<Particle> &ghost_particles) const
{
  Direction u_lab {det_pos[0]-p.r().x,  // towards the detector
                   det_pos[1]-p.r().y,
                   det_pos[2]-p.r().z};
  Direction u_lab_unit = u_lab/u_lab.norm(); // normalize
  // Sample cosine of scattering angle
  double pdf = -1;
  if (!angle_.empty()) {
    mymu = angle_.sample(E_in, seed);
    pdf = angle_.get_pdf(E_in,mymu,seed);
  } else {
    // no angle distribution given => assume isotropic for all energies
    mymu = uniform_distribution(-1., 1., seed);
    pdf = 0.5;
  }

  // Sample outgoing energy
  E_out = energy_->sample(E_in, seed);

 const auto& nuc {data::nuclides[p.event_nuclide()]};
 const auto& rx {nuc->reactions_[p.event_index_mt()]};

   if (rx->scatter_in_cm_) {

    //std::cout << " COM scatter "  <<std::endl;
    double E_cm = E_out;

    // determine outgoing energy in lab
    double A = nuc->awr_;
    double E_lab = E_cm + (E_in + 2.0 * mymu * (A + 1.0) * std::sqrt(E_in * E_cm)) /
                 ((A + 1.0) * (A + 1.0));

    // determine outgoing angle in lab
    mymu = mymu * std::sqrt(E_cm / E_lab) + 1.0 / (A + 1.0) * std::sqrt(E_in / E_lab);
   

    Particle ghost_particle=Particle();
    ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab);
    ghost_particles.push_back(ghost_particle);
    pdfs_cm.push_back(pdf);
    double deriv = sqrt(E_lab / E_out) /(1 - mymu / (A + 1) * sqrt(E_in /E_lab));
    double pdf_mu_lab = pdf * deriv;
    pdfs_lab.push_back(pdf_mu_lab);
   // std::cout << " E in "  <<E_in << std::endl;
    //std::cout << " E lab "  <<E_lab << std::endl;
    //std::cout << " pdf lab "  <<(pdf_mu_lab) << std::endl;

  }
  else
  {
     std::cout << " LAB scatter "  <<std::endl;
    Particle ghost_particle=Particle();
    ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_out);
    ghost_particles.push_back(ghost_particle);
    pdfs_cm.push_back(-999);
    pdfs_lab.push_back(pdf);
    std::cout << " pdf lab "  <<pdf << std::endl;
    //std::cout << " E lab "  <<E_out<< std::endl;

  }
 


}



} // namespace openmc
