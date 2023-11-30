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
  double det_pos[3],double E_in,double& E_out, uint64_t* seed , Particle &p,std::vector<double> &pdfs_cm , std::vector<double> &pdfs_lab ,std::vector<Particle> &ghost_particles) const
{
  const auto& nuc {data::nuclides[p.event_nuclide()]};
  const auto& rx {nuc->reactions_[p.event_index_mt()]};
  double A = nuc->awr_;
  Direction u_lab {det_pos[0]-p.r().x,  // towards the detector
                   det_pos[1]-p.r().y,
                   det_pos[2]-p.r().z};
  Direction u_lab_unit = u_lab/u_lab.norm(); // normalize
  // Sample cosine of scattering angle
  double m1= p.getMass()/1e6; // mass of incoming particle in MeV
  double m2= m1*A; // mass of target 
  double E1_tot = p.E_last()/1e6 + m1; // total Energy of incoming particle in MeV
  double p1_tot = std::sqrt(E1_tot*E1_tot  - m1*m1); // total momenta of incoming particle in MeV
  Direction p1=p1_tot*p.u_last(); // 3 momentum of incoming particle
  Direction p2= p.v_t() * m2 /C_LIGHT; //3 momentum of target in lab 
  double E2_tot = std::sqrt(p2.norm()*p2.norm() + m2*m2); // 
  double E_cm = E1_tot + E2_tot;
  Direction p_cm = p1 + p2;
  double p_tot_cm = p_cm.norm();
  double mu_lab = u_lab_unit.dot(p_cm) /  ( p_tot_cm ) ;  // between cm and p3





  

  // Sample outgoing energy
  E_out = energy_->sample(E_in, seed);

   if (rx->scatter_in_cm_) {
    //std::cout << " COM scatter "  <<std::endl;
    double cond =E_out * (A+1)*(A+1) + E_in * (mu_lab*mu_lab - 1);
   if ( cond >= 0)
   {
    double  E_lab1 = (E_out * (A+1)*(A+1) - 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1)) / ((A+1)*(A+1));
    double  mu_cm1 =  (mu_lab - 1/(A+1) * std::sqrt(E_in/E_lab1))*std::sqrt(E_lab1/E_out);
     double pdf_cm1 = -1;
   if (!angle_.empty()) {
    pdf_cm1 = angle_.get_pdf(E_in,mu_cm1,seed);
  } else {
    // no angle distribution given => assume isotropic for all energies
    pdf_cm1 = 0.5;
  }
  double E_lab1_maybe = E_out + (E_in + 2.0 * mu_cm1 * (A + 1.0) * std::sqrt(E_in * E_out)) /((A + 1.0) * (A + 1.0));
     // std::cout << "E_lab1 maybe? " << E_lab1_maybe  << std::endl;

      double E1_lab_diff = std::abs(E_lab1 - E_lab1_maybe);
     // std::cout << "E_lab1 diff " << E1_lab_diff << std::endl;

 if (E1_lab_diff<0.01)
   {
    
    
    Particle ghost_particle=Particle();
    ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab1);
    ghost_particles.push_back(ghost_particle);
    pdfs_cm.push_back(pdf_cm1);
    double deriv = sqrt(E_lab1 / E_out) /(1 - mu_lab / (A + 1) * sqrt(E_in /E_lab1));
    double pdf_mu1_lab = pdf_cm1 * deriv;
    pdfs_lab.push_back(pdf_mu1_lab);
   }
   // std::cout << "pdf_mu1_lab " << pdf_mu1_lab << std::endl;
  // std::cout << "E_lab1: " << E_lab1 << std::endl;
  //  std::cout << "mu_cm1: " << mu_cm1 << std::endl;
  //  std::cout << "pdf_mu1_cm: " << pdf_cm1 << std::endl;

    if (cond > 0)
    {
      double  E_lab2 = (E_out * (A+1)*(A+1) + 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1)) / ((A+1)*(A+1));
      double  mu_cm2 =  (mu_lab - 1/(A+1) * std::sqrt(E_in/E_lab2))*std::sqrt(E_lab2/E_out);
      double pdf_cm2 = -1;
   if (!angle_.empty()) {
    pdf_cm2 = angle_.get_pdf(E_in,mu_cm2,seed);
  } else {
    // no angle distribution given => assume isotropic for all energies
    pdf_cm2 = 0.5;
  }

  double E_lab2_maybe = E_out + (E_in + 2.0 * mu_cm2 * (A + 1.0) * std::sqrt(E_in * E_out)) /((A + 1.0) * (A + 1.0));
     // std::cout << "E_lab2 maybe? " << E_lab2_maybe  << std::endl;

  double E2_lab_diff = std::abs(E_lab2 - E_lab2_maybe);
     // std::cout << "E_lab2 diff " << E2_lab_diff << std::endl;

 if (E2_lab_diff<0.01)
   {
      
      Particle ghost_particle=Particle();
      ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab2);
      ghost_particles.push_back(ghost_particle);
      pdfs_cm.push_back(pdf_cm2);
      double deriv = sqrt(E_lab2 / E_out) /(1 - mu_lab / (A + 1) * sqrt(E_in /E_lab2));
      double pdf_mu2_lab = pdf_cm2 * deriv;
      pdfs_lab.push_back(pdf_mu2_lab);
     // std::cout << "pdf_mu2_lab " << pdf_mu2_lab << std::endl;
   //  std::cout << "E_lab2: " << E_lab2 << std::endl;
   // std::cout << "mu_cm2: " << mu_cm2 << std::endl;
   // std::cout << "pdf_mu2_cm: " << pdf_cm2 << std::endl;
   }
    }

   }


   

  }
 
 if (!rx->scatter_in_cm_)
   {
    //finding mu_cm, E_out is in lab
  double E_lab = E_out;
  Particle ghost_particle=Particle();
  ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab);
  ghost_particles.push_back(ghost_particle);
  pdfs_cm.push_back(-999);
  double pdf_mu_lab;
  if (!angle_.empty()) {
    pdf_mu_lab = angle_.get_pdf(E_in,mu_lab,seed);
  } else {
    // no angle distribution given => assume isotropic for all energies
    pdf_mu_lab = 0.5;
  }

  pdfs_lab.push_back(pdf_mu_lab);
 
   }

}



} // namespace openmc
