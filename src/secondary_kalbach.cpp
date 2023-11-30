#include "openmc/secondary_kalbach.h"

#include <algorithm> // for copy, move
#include <cmath>     // for log, sqrt, sinh
#include <cstddef>   // for size_t
#include <iterator>  // for back_inserter

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include "openmc/hdf5_interface.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/vector.h"
#include "openmc/particle.h"
#include "openmc/nuclide.h"

namespace openmc {

//==============================================================================
//! KalbachMann implementation
//==============================================================================

KalbachMann::KalbachMann(hid_t group)
{
  // Open incoming energy dataset
  hid_t dset = open_dataset(group, "energy");

  // Get interpolation parameters
  xt::xarray<int> temp;
  read_attribute(dset, "interpolation", temp);

  auto temp_b = xt::view(temp, 0); // view of breakpoints
  auto temp_i = xt::view(temp, 1); // view of interpolation parameters

  std::copy(temp_b.begin(), temp_b.end(), std::back_inserter(breakpoints_));
  for (const auto i : temp_i)
    interpolation_.push_back(int2interp(i));
  n_region_ = breakpoints_.size();

  // Get incoming energies
  read_dataset(dset, energy_);
  std::size_t n_energy = energy_.size();
  close_dataset(dset);

  // Get outgoing energy distribution data
  dset = open_dataset(group, "distribution");
  vector<int> offsets;
  vector<int> interp;
  vector<int> n_discrete;
  read_attribute(dset, "offsets", offsets);
  read_attribute(dset, "interpolation", interp);
  read_attribute(dset, "n_discrete_lines", n_discrete);

  xt::xarray<double> eout;
  read_dataset(dset, eout);
  close_dataset(dset);

  for (int i = 0; i < n_energy; ++i) {
    // Determine number of outgoing energies
    int j = offsets[i];
    int n;
    if (i < n_energy - 1) {
      n = offsets[i + 1] - j;
    } else {
      n = eout.shape()[1] - j;
    }

    // Assign interpolation scheme and number of discrete lines
    KMTable d;
    d.interpolation = int2interp(interp[i]);
    d.n_discrete = n_discrete[i];

    // Copy data
    d.e_out = xt::view(eout, 0, xt::range(j, j + n));
    d.p = xt::view(eout, 1, xt::range(j, j + n));
    d.c = xt::view(eout, 2, xt::range(j, j + n));
    d.r = xt::view(eout, 3, xt::range(j, j + n));
    d.a = xt::view(eout, 4, xt::range(j, j + n));

    // To get answers that match ACE data, for now we still use the tabulated
    // CDF values that were passed through to the HDF5 library. At a later
    // time, we can remove the CDF values from the HDF5 library and
    // reconstruct them using the PDF
    if (false) {
      // Calculate cumulative distribution function -- discrete portion
      for (int k = 0; k < d.n_discrete; ++k) {
        if (k == 0) {
          d.c[k] = d.p[k];
        } else {
          d.c[k] = d.c[k - 1] + d.p[k];
        }
      }

      // Continuous portion
      for (int k = d.n_discrete; k < n; ++k) {
        if (k == d.n_discrete) {
          d.c[k] = d.c[k - 1] + d.p[k];
        } else {
          if (d.interpolation == Interpolation::histogram) {
            d.c[k] = d.c[k - 1] + d.p[k - 1] * (d.e_out[k] - d.e_out[k - 1]);
          } else if (d.interpolation == Interpolation::lin_lin) {
            d.c[k] = d.c[k - 1] + 0.5 * (d.p[k - 1] + d.p[k]) *
                                    (d.e_out[k] - d.e_out[k - 1]);
          }
        }
      }

      // Normalize density and distribution functions
      d.p /= d.c[n - 1];
      d.c /= d.c[n - 1];
    }

    distribution_.push_back(std::move(d));
  } // incoming energies
}

void KalbachMann::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  auto n_energy_in = energy_.size();
  int i;
  double r;
  if (E_in < energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (E_in > energy_[n_energy_in - 1]) {
    i = n_energy_in - 2;
    r = 1.0;
  } else {
    i = lower_bound_index(energy_.begin(), energy_.end(), E_in);
    r = (E_in - energy_[i]) / (energy_[i + 1] - energy_[i]);
  }

  // Sample between the ith and [i+1]th bin
  int l = r > prn(seed) ? i + 1 : i;

  // Interpolation for energy E1 and EK
  int n_energy_out = distribution_[i].e_out.size();
  int n_discrete = distribution_[i].n_discrete;
  double E_i_1 = distribution_[i].e_out[n_discrete];
  double E_i_K = distribution_[i].e_out[n_energy_out - 1];

  n_energy_out = distribution_[i + 1].e_out.size();
  n_discrete = distribution_[i + 1].n_discrete;
  double E_i1_1 = distribution_[i + 1].e_out[n_discrete];
  double E_i1_K = distribution_[i + 1].e_out[n_energy_out - 1];

  double E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
  double E_K = E_i_K + r * (E_i1_K - E_i_K);

  // Determine outgoing energy bin
  n_energy_out = distribution_[l].e_out.size();
  n_discrete = distribution_[l].n_discrete;
  double r1 = prn(seed);
  double c_k = distribution_[l].c[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  for (int j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = distribution_[l].c[k];
    if (r1 < c_k) {
      end = j;
      break;
    }
  }

  // Continuous portion
  double c_k1;
  for (int j = n_discrete; j < end; ++j) {
    k = j;
    c_k1 = distribution_[l].c[k + 1];
    if (r1 < c_k1)
      break;
    k = j + 1;
    c_k = c_k1;
  }

  double E_l_k = distribution_[l].e_out[k];
  double p_l_k = distribution_[l].p[k];
  double km_r, km_a;
  if (distribution_[l].interpolation == Interpolation::histogram) {
    // Histogram interpolation
    if (p_l_k > 0.0 && k >= n_discrete) {
      E_out = E_l_k + (r1 - c_k) / p_l_k;
    } else {
      E_out = E_l_k;
    }

    // Determine Kalbach-Mann parameters
    km_r = distribution_[l].r[k];
    km_a = distribution_[l].a[k];

  } else {
    // Linear-linear interpolation
    double E_l_k1 = distribution_[l].e_out[k + 1];
    double p_l_k1 = distribution_[l].p[k + 1];

    double frac = (p_l_k1 - p_l_k) / (E_l_k1 - E_l_k);
    if (frac == 0.0) {
      E_out = E_l_k + (r1 - c_k) / p_l_k;
    } else {
      E_out =
        E_l_k +
        (std::sqrt(std::max(0.0, p_l_k * p_l_k + 2.0 * frac * (r1 - c_k))) -
          p_l_k) /
          frac;
    }

    // Determine Kalbach-Mann parameters
    km_r = distribution_[l].r[k] +
           (E_out - E_l_k) / (E_l_k1 - E_l_k) *
             (distribution_[l].r[k + 1] - distribution_[l].r[k]);
    km_a = distribution_[l].a[k] +
           (E_out - E_l_k) / (E_l_k1 - E_l_k) *
             (distribution_[l].a[k + 1] - distribution_[l].a[k]);
  }

  // Now interpolate between incident energy bins i and i + 1
  if (k >= n_discrete) {
    if (l == i) {
      E_out = E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1);
    } else {
      E_out = E_1 + (E_out - E_i1_1) * (E_K - E_1) / (E_i1_K - E_i1_1);
    }
  }

  // Sampled correlated angle from Kalbach-Mann parameters
  if (prn(seed) > km_r) {
    double T = uniform_distribution(-1., 1., seed) * std::sinh(km_a);
    mu = std::log(T + std::sqrt(T * T + 1.0)) / km_a;
  } else {
    double r1 = prn(seed);
    mu = std::log(r1 * std::exp(km_a) + (1.0 - r1) * std::exp(-km_a)) / km_a;
  }
}
void KalbachMann::get_pdf(double det_pos[3] , double E_in,double& E_out, uint64_t* seed, Particle &p ,std::vector<double> &pdfs_cm , std::vector<double> &pdfs_lab ,std::vector<Particle> &ghost_particles) const
{
  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  //double mu_lab = mymu; // get lab value from tally scoring

  
  double myE_in = E_in;

  const auto& nuc {data::nuclides[p.event_nuclide()]};
  double A = nuc->awr_;
  Direction u_lab {det_pos[0]-p.r().x,  // towards the detector
                   det_pos[1]-p.r().y,
                   det_pos[2]-p.r().z};
  Direction u_lab_unit = u_lab/u_lab.norm(); // normalize
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





  //double E_out;
  auto n_energy_in = energy_.size();
  int i;
  double r;
  if (myE_in < energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (myE_in > energy_[n_energy_in - 1]) {
    i = n_energy_in - 2;
    r = 1.0;
  } else {
    i = lower_bound_index(energy_.begin(), energy_.end(), myE_in);
    r = (myE_in - energy_[i]) / (energy_[i + 1] - energy_[i]);
  }

  // Sample between the ith and [i+1]th bin
  int l = r > prn(seed) ? i + 1 : i;

  // Interpolation for energy E1 and EK
  int n_energy_out = distribution_[i].e_out.size();
  int n_discrete = distribution_[i].n_discrete;
  double E_i_1 = distribution_[i].e_out[n_discrete];
  double E_i_K = distribution_[i].e_out[n_energy_out - 1];

  n_energy_out = distribution_[i + 1].e_out.size();
  n_discrete = distribution_[i + 1].n_discrete;
  double E_i1_1 = distribution_[i + 1].e_out[n_discrete];
  double E_i1_K = distribution_[i + 1].e_out[n_energy_out - 1];

  double E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
  double E_K = E_i_K + r * (E_i1_K - E_i_K);

  // Determine outgoing energy bin
  n_energy_out = distribution_[l].e_out.size();
  n_discrete = distribution_[l].n_discrete;
  double r1 = prn(seed);
  double c_k = distribution_[l].c[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  for (int j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = distribution_[l].c[k];
    if (r1 < c_k) {
      end = j;
      break;
    }
  }

  // Continuous portion
  double c_k1;
  for (int j = n_discrete; j < end; ++j) {
    k = j;
    c_k1 = distribution_[l].c[k + 1];
    if (r1 < c_k1)
      break;
    k = j + 1;
    c_k = c_k1;
  }

  double E_l_k = distribution_[l].e_out[k];
  double p_l_k = distribution_[l].p[k];
  double km_r, km_a;
  if (distribution_[l].interpolation == Interpolation::histogram) {
    // Histogram interpolation
    if (p_l_k > 0.0 && k >= n_discrete) {
      E_out = E_l_k + (r1 - c_k) / p_l_k;
    } else {
      E_out = E_l_k;
    }
   //  std::cout << " Histogram interpolation" <<std::endl;
    // Determine Kalbach-Mann parameters
    km_r = distribution_[l].r[k];
    km_a = distribution_[l].a[k];
    //std::cout <<"Histo " << " km_r " <<  km_r << " km_a " <<  km_a << std::endl;

  } else {
    // Linear-linear interpolation
    double E_l_k1 = distribution_[l].e_out[k + 1];
    double p_l_k1 = distribution_[l].p[k + 1];

    double frac = (p_l_k1 - p_l_k) / (E_l_k1 - E_l_k);
    if (frac == 0.0) {
      E_out = E_l_k + (r1 - c_k) / p_l_k;
    } else {
      E_out =
        E_l_k +
        (std::sqrt(std::max(0.0, p_l_k * p_l_k + 2.0 * frac * (r1 - c_k))) -
          p_l_k) /
          frac;
    }
 //  std::cout << " Linear-linear interpolation " <<std::endl;
    // Determine Kalbach-Mann parameters
    km_r = distribution_[l].r[k] +
           (E_out - E_l_k) / (E_l_k1 - E_l_k) *
             (distribution_[l].r[k + 1] - distribution_[l].r[k]);
    km_a = distribution_[l].a[k] +
           (E_out - E_l_k) / (E_l_k1 - E_l_k) *
             (distribution_[l].a[k + 1] - distribution_[l].a[k]);
  }


// Now interpolate between incident energy bins i and i + 1
  if (k >= n_discrete) {
    if (l == i) {
      E_out = E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1);
    } else {
      E_out = E_1 + (E_out - E_i1_1) * (E_K - E_1) / (E_i1_K - E_i1_1);
    }
  }
  //std::cout << " my E out in KM  " << E_out <<std::endl;
  double cond =E_out * (A+1)*(A+1) + E_in * (mu_lab*mu_lab - 1);
   if ( cond >= 0)
   {
    double  E_lab1 = (E_out * (A+1)*(A+1) - 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1)) / ((A+1)*(A+1));
    double  mu_cm1 =  (mu_lab - 1/(A+1) * std::sqrt(E_in/E_lab1))*std::sqrt(E_lab1/E_out);
     double pdf_mu1_cm = km_a / (2 * std::sinh(km_a)) * (std::cosh(km_a * mu_cm1) + km_r * std::sinh(km_a * mu_cm1)); // center of mass
     /*
     std::cout << "cond " << cond<< ","<<std::endl;
     std::cout << "ch " << - 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1) << ","<<std::endl;
     std::cout << "A: " << A<< ","<<std::endl;
     std::cout << "E_cm_out: " << E_out<< "," << std::endl;
      std::cout << "mu_lab: " << mu_lab<< "," << std::endl;
      std::cout << "E: " << E_in << std::endl;
      */
   //std::cout << "E_lab1: " << E_lab1 << std::endl;
   //std::cout << "mu_cm1: " << mu_cm1 << std::endl;
   

   //std::cout << "mu_cm1 recons: " << mu_cm1 << std::endl;
     double E_lab1_maybe = E_out + (E_in + 2.0 * mu_cm1 * (A + 1.0) * std::sqrt(E_in * E_out)) /((A + 1.0) * (A + 1.0));
    // std::cout << "E_lab1 maybe? " << E_lab1_maybe  << std::endl;
   double E1_lab_diff = std::abs(E_lab1 - E_lab1_maybe);
     // std::cout << "E_lab1 diff " << E1_lab_diff << std::endl;

   // std::cout << "pdf_mu1_cm: " << pdf_mu1_cm << std::endl;
   if (E1_lab_diff<0.01)
   {

    Particle ghost_particle=Particle();
    ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab1);
    ghost_particles.push_back(ghost_particle);
    pdfs_cm.push_back(pdf_mu1_cm);
    double deriv = sqrt(E_lab1 / E_out) /(1 - mu_lab / (A + 1) * sqrt(E_in /E_lab1));
    double pdf_mu1_lab = pdf_mu1_cm * deriv;
    pdfs_lab.push_back(pdf_mu1_lab);
   // std::cout << "pdf_mu1_lab " << pdf_mu1_lab << std::endl;
   }

    if (cond > 0)
    {
      double  E_lab2 = (E_out * (A+1)*(A+1) + 2 * std::sqrt(E_in) * mu_lab * std::sqrt(cond) + E_in * (2 * mu_lab*mu_lab - 1)) / ((A+1)*(A+1));
      double  mu_cm2 =  (mu_lab - 1/(A+1) * std::sqrt(E_in/E_lab2))*std::sqrt(E_lab2/E_out);
      double pdf_mu2_cm = km_a / (2 * std::sinh(km_a)) * (std::cosh(km_a * mu_cm2) + km_r * std::sinh(km_a * mu_cm2)); // center of mass
      //std::cout << "E_lab2: " << E_lab2 << std::endl;
     // std::cout << "mu_cm2: " << mu_cm2 << std::endl;
   // std::cout << "pdf_mu2_cm: " << pdf_mu2_cm << std::endl;
     double E_lab2_maybe = E_out + (E_in + 2.0 * mu_cm2 * (A + 1.0) * std::sqrt(E_in * E_out)) /((A + 1.0) * (A + 1.0));
     // std::cout << "E_lab2 maybe? " << E_lab2_maybe  << std::endl;

      double E2_lab_diff = std::abs(E_lab2 - E_lab2_maybe);
     // std::cout << "E_lab2 diff " << E2_lab_diff << std::endl;

 if (E2_lab_diff<0.01)
   {

      Particle ghost_particle=Particle();
      ghost_particle.initilze_ghost_particle(p,u_lab_unit,E_lab2);
      ghost_particles.push_back(ghost_particle);
      pdfs_cm.push_back(pdf_mu2_cm);
      double deriv = sqrt(E_lab2 / E_out) /(1 - mu_lab / (A + 1) * sqrt(E_in /E_lab2));
      double pdf_mu2_lab = pdf_mu2_cm * deriv;
      pdfs_lab.push_back(pdf_mu2_lab);
     // std::cout << "pdf_mu2_lab " << pdf_mu2_lab << std::endl;
   }
    
    }

   }
// https://docs.openmc.org/en/v0.8.0/methods/physics.html#equation-KM-pdf-angle
   //double pdf_mu = km_a / (2 * std::sinh(km_a)) * (std::cosh(km_a * mymu) + km_r * std::sinh(km_a * mymu)); // center of mass
   //return pdf_mu;

const auto& rx {nuc->reactions_[p.event_index_mt()]};
    if (!rx->scatter_in_cm_)
   {
   fatal_error("didn't implement lab");
   }
}

} // namespace openmc
