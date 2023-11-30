#include "openmc/secondary_nbody.h"

#include <cmath> // for log

#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/nuclide.h"

namespace openmc {

//==============================================================================
// NBodyPhaseSpace implementation
//==============================================================================

NBodyPhaseSpace::NBodyPhaseSpace(hid_t group)
{
  read_attribute(group, "n_particles", n_bodies_);
  read_attribute(group, "total_mass", mass_ratio_);
  read_attribute(group, "atomic_weight_ratio", A_);
  read_attribute(group, "q_value", Q_);
}

void NBodyPhaseSpace::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  // By definition, the distribution of the angle is isotropic for an N-body
  // phase space distribution
  mu = uniform_distribution(-1., 1., seed);

  // Determine E_max parameter
  double Ap = mass_ratio_;
  double E_max = (Ap - 1.0) / Ap * (A_ / (A_ + 1.0) * E_in + Q_);

  // x is essentially a Maxwellian distribution
  double x = maxwell_spectrum(1.0, seed);

  double y;
  double r1, r2, r3, r4, r5, r6;
  switch (n_bodies_) {
  case 3:
    y = maxwell_spectrum(1.0, seed);
    break;
  case 4:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    y = -std::log(r1 * r2 * r3);
    break;
  case 5:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    r4 = prn(seed);
    r5 = prn(seed);
    r6 = prn(seed);
    y = -std::log(r1 * r2 * r3 * r4) -
        std::log(r5) * std::pow(std::cos(PI / 2.0 * r6), 2);
    break;
  default:
    throw std::runtime_error {"N-body phase space with >5 bodies."};
  }

  // Now determine v and E_out
  double v = x / (x + y);
  E_out = E_max * v;
}

void NBodyPhaseSpace::get_pdf(
  double det_pos[3],double E_in,double& E_out,double mymu, uint64_t* seed , Particle &p,std::vector<double> &pdfs_cm , std::vector<double> &pdfs_lab ,std::vector<Particle> &ghost_particles) const
{
// By definition, the distribution of the angle is isotropic for an N-body
  // phase space distribution
  mymu = uniform_distribution(-1., 1., seed);

  // Determine E_max parameter
  double Ap = mass_ratio_;
  double E_max = (Ap - 1.0) / Ap * (A_ / (A_ + 1.0) * E_in + Q_);

  // x is essentially a Maxwellian distribution
  double x = maxwell_spectrum(1.0, seed);

  double y;
  double r1, r2, r3, r4, r5, r6;
  switch (n_bodies_) {
  case 3:
    y = maxwell_spectrum(1.0, seed);
    break;
  case 4:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    y = -std::log(r1 * r2 * r3);
    break;
  case 5:
    r1 = prn(seed);
    r2 = prn(seed);
    r3 = prn(seed);
    r4 = prn(seed);
    r5 = prn(seed);
    r6 = prn(seed);
    y = -std::log(r1 * r2 * r3 * r4) -
        std::log(r5) * std::pow(std::cos(PI / 2.0 * r6), 2);
    break;
  default:
    throw std::runtime_error {"N-body phase space with >5 bodies."};
  }

  // Now determine v and E_out
  double v = x / (x + y);
  E_out = E_max * v;




const auto& nuc {data::nuclides[p.event_nuclide()]};
const auto& rx {nuc->reactions_[p.event_index_mt()]};
 Direction u_lab {det_pos[0]-p.r().x,  // towards the detector
                   det_pos[1]-p.r().y,
                   det_pos[2]-p.r().z};
Direction u_lab_unit = u_lab/u_lab.norm(); // normalize

   if (rx->scatter_in_cm_) {
   // std::cout << "E_out_cm in nbody" << E_out << std::endl;
    // determine outgoing energy in lab
    double E_cm = E_out;
    double A = nuc->awr_;
    double E_lab = E_cm + (E_in + 2.0 * mymu * (A + 1.0) * std::sqrt(E_in * E_cm)) /
                 ((A + 1.0) * (A + 1.0));
    // determine outgoing angle in lab
    mymu = mymu * std::sqrt(E_cm / E_lab) + 1.0 / (A + 1.0) * std::sqrt(E_in / E_lab);
    Particle ghost_particle=Particle();
    ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab);
    ghost_particles.push_back(ghost_particle);
    pdfs_cm.push_back(0.5);
    double deriv = sqrt(E_lab / E_out) /(1 - mymu / (A + 1) * sqrt(E_in /E_lab));
    double pdf_mu_lab = 0.5 * deriv;
    pdfs_lab.push_back(pdf_mu_lab);   
    //std::cout << "E_out_lab in nbody" << E_lab << std::endl;         



   }

   else
   {
   fatal_error("didn't implement lab");
   }


}



} // namespace openmc
