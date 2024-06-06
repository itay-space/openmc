#include "openmc/thermal.h"
#include <iomanip>
#include <algorithm> // for sort, move, min, max, find
#include <cmath>     // for round, sqrt, abs
#include <fstream>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include <fmt/core.h>

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/error.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/secondary_correlated.h"
#include "openmc/secondary_thermal.h"
#include "openmc/settings.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace data {
std::unordered_map<std::string, int> thermal_scatt_map;
vector<unique_ptr<ThermalScattering>> thermal_scatt;
} // namespace data

//==============================================================================
// ThermalScattering implementation
//==============================================================================

ThermalScattering::ThermalScattering(
  hid_t group, const vector<double>& temperature)
{
  // Get name of table from group
  name_ = object_name(group);

  // Get rid of leading '/'
  name_ = name_.substr(1);

  read_attribute(group, "atomic_weight_ratio", awr_);
  read_attribute(group, "energy_max", energy_max_);
  read_attribute(group, "nuclides", nuclides_);

  // Read temperatures
  hid_t kT_group = open_group(group, "kTs");

  // Determine temperatures available
  auto dset_names = dataset_names(kT_group);
  auto n = dset_names.size();
  auto temps_available = xt::empty<double>({n});
  for (int i = 0; i < dset_names.size(); ++i) {
    // Read temperature value
    double T;
    read_dataset(kT_group, dset_names[i].data(), T);
    temps_available[i] = T / K_BOLTZMANN;
  }
  std::sort(temps_available.begin(), temps_available.end());

  // Determine actual temperatures to read -- start by checking whether a
  // temperature range was given, in which case all temperatures in the range
  // are loaded irrespective of what temperatures actually appear in the model
  vector<int> temps_to_read;
  if (settings::temperature_range[1] > 0.0) {
    for (const auto& T : temps_available) {
      if (settings::temperature_range[0] <= T &&
          T <= settings::temperature_range[1]) {
        temps_to_read.push_back(std::round(T));
      }
    }
  }

  switch (settings::temperature_method) {
  case TemperatureMethod::NEAREST:
    // Determine actual temperatures to read
    for (const auto& T : temperature) {

      auto i_closest = xt::argmin(xt::abs(temps_available - T))[0];
      auto temp_actual = temps_available[i_closest];
      if (std::abs(temp_actual - T) < settings::temperature_tolerance) {
        if (std::find(temps_to_read.begin(), temps_to_read.end(),
              std::round(temp_actual)) == temps_to_read.end()) {
          temps_to_read.push_back(std::round(temp_actual));
        }
      } else {
        fatal_error(fmt::format("Nuclear data library does not contain cross "
                                "sections for {} at or near {} K.",
          name_, std::round(T)));
      }
    }
    break;

  case TemperatureMethod::INTERPOLATION:
    // If temperature interpolation or multipole is selected, get a list of
    // bounding temperatures for each actual temperature present in the model
    for (const auto& T : temperature) {
      bool found = false;
      for (int j = 0; j < temps_available.size() - 1; ++j) {
        if (temps_available[j] <= T && T < temps_available[j + 1]) {
          int T_j = std::round(temps_available[j]);
          int T_j1 = std::round(temps_available[j + 1]);
          if (std::find(temps_to_read.begin(), temps_to_read.end(), T_j) ==
              temps_to_read.end()) {
            temps_to_read.push_back(T_j);
          }
          if (std::find(temps_to_read.begin(), temps_to_read.end(), T_j1) ==
              temps_to_read.end()) {
            temps_to_read.push_back(T_j1);
          }
          found = true;
        }
      }
      if (!found) {
        // If no pairs found, check if the desired temperature falls within
        // bounds' tolerance
        if (std::abs(T - temps_available[0]) <=
            settings::temperature_tolerance) {
          if (std::find(temps_to_read.begin(), temps_to_read.end(),
                std::round(temps_available[0])) == temps_to_read.end()) {
            temps_to_read.push_back(std::round(temps_available[0]));
          }
        } else if (std::abs(T - temps_available[n - 1]) <=
                   settings::temperature_tolerance) {
          if (std::find(temps_to_read.begin(), temps_to_read.end(),
                std::round(temps_available[n - 1])) == temps_to_read.end()) {
            temps_to_read.push_back(std::round(temps_available[n - 1]));
          }
        } else {
          fatal_error(
            fmt::format("Nuclear data library does not contain cross "
                        "sections for {} at temperatures that bound {} K.",
              name_, std::round(T)));
        }
      }
    }
  }

  // Sort temperatures to read
  std::sort(temps_to_read.begin(), temps_to_read.end());

  auto n_temperature = temps_to_read.size();
  kTs_.reserve(n_temperature);
  data_.reserve(n_temperature);

  for (auto T : temps_to_read) {
    // Get temperature as a string
    std::string temp_str = fmt::format("{}K", T);

    // Read exact temperature value
    double kT;
    read_dataset(kT_group, temp_str.data(), kT);
    kTs_.push_back(kT);

    // Open group for this temperature
    hid_t T_group = open_group(group, temp_str.data());
    data_.emplace_back(T_group);
    close_group(T_group);
  }

  close_group(kT_group);
}

void ThermalScattering::calculate_xs(double E, double sqrtkT, int* i_temp,
  double* elastic, double* inelastic, uint64_t* seed) const
{
  // Determine temperature for S(a,b) table
  double kT = sqrtkT * sqrtkT;
  int i = 0;

  auto n = kTs_.size();
  if (n > 1) {
    if (settings::temperature_method == TemperatureMethod::NEAREST) {
      while (kTs_[i + 1] < kT && i + 1 < n - 1)
        ++i;
      // Pick closer of two bounding temperatures
      if (kT - kTs_[i] > kTs_[i + 1] - kT)
        ++i;
    } else {
      // If current kT outside of the bounds of available, snap to the bound
      if (kT < kTs_.front()) {
        i = 0;
      } else if (kT > kTs_.back()) {
        i = kTs_.size() - 1;
      } else {
        // Find temperatures that bound the actual temperature
        while (kTs_[i + 1] < kT && i + 1 < n - 1)
          ++i;
        // Randomly sample between temperature i and i+1
        double f = (kT - kTs_[i]) / (kTs_[i + 1] - kTs_[i]);
        if (f > prn(seed))
          ++i;
      }
    }
  }

  // Set temperature index
  *i_temp = i;

  // Calculate cross sections for ith temperature
  data_[i].calculate_xs(E, elastic, inelastic);
}

bool ThermalScattering::has_nuclide(const char* name) const
{
  std::string nuc {name};
  return std::find(nuclides_.begin(), nuclides_.end(), nuc) != nuclides_.end();
}

//==============================================================================
// ThermalData implementation
//==============================================================================

ThermalData::ThermalData(hid_t group)
{
  // Coherent/incoherent elastic data
  if (object_exists(group, "elastic")) {
    // Read cross section data
    hid_t elastic_group = open_group(group, "elastic");

    // Read elastic cross section
    elastic_.xs = read_function(elastic_group, "xs");

    // Read angle-energy distribution
    hid_t dgroup = open_group(elastic_group, "distribution");
    std::string temp;
    read_attribute(dgroup, "type", temp);
    if (temp == "coherent_elastic") {
      auto xs = dynamic_cast<CoherentElasticXS*>(elastic_.xs.get());
      elastic_.distribution = make_unique<CoherentElasticAE>(*xs);
    } else if (temp == "incoherent_elastic") {
      elastic_.distribution = make_unique<IncoherentElasticAE>(dgroup);
    } else if (temp == "incoherent_elastic_discrete") {
      auto xs = dynamic_cast<Tabulated1D*>(elastic_.xs.get());
      elastic_.distribution =
        make_unique<IncoherentElasticAEDiscrete>(dgroup, xs->x());
    } else if (temp == "mixed_elastic") {
      // Get coherent/incoherent cross sections
      auto mixed_xs = dynamic_cast<Sum1D*>(elastic_.xs.get());
      const auto& coh_xs =
        dynamic_cast<const CoherentElasticXS*>(mixed_xs->functions(0).get());
      const auto& incoh_xs = mixed_xs->functions(1).get();

      // Create mixed elastic distribution
      elastic_.distribution =
        make_unique<MixedElasticAE>(dgroup, *coh_xs, *incoh_xs);
    }

    close_group(elastic_group);
  }

  // Inelastic data
  if (object_exists(group, "inelastic")) {
    // Read type of inelastic data
    hid_t inelastic_group = open_group(group, "inelastic");

    // Read inelastic cross section
    inelastic_.xs = read_function(inelastic_group, "xs");

    // Read angle-energy distribution
    hid_t dgroup = open_group(inelastic_group, "distribution");
    std::string temp;
    read_attribute(dgroup, "type", temp);
    if (temp == "incoherent_inelastic") {
      inelastic_.distribution = make_unique<IncoherentInelasticAE>(dgroup);
    } else if (temp == "incoherent_inelastic_discrete") {
      auto xs = dynamic_cast<Tabulated1D*>(inelastic_.xs.get());
      inelastic_.distribution =
        make_unique<IncoherentInelasticAEDiscrete>(dgroup, xs->x());
    }

    close_group(inelastic_group);
  }
}

void ThermalData::calculate_xs(
  double E, double* elastic, double* inelastic) const
{
  // Calculate thermal elastic scattering cross section
  if (elastic_.xs) {
    *elastic = (*elastic_.xs)(E);
  } else {
    *elastic = 0.0;
  }

  // Calculate thermal inelastic scattering cross section
  *inelastic = (*inelastic_.xs)(E);
}

void ThermalData::sample(const NuclideMicroXS& micro_xs, double E,
  double* E_out, double* mu, uint64_t* seed)
{
  // Determine whether inelastic or elastic scattering will occur
  if (prn(seed) < micro_xs.thermal_elastic / micro_xs.thermal) {
    elastic_.distribution->sample(E, *E_out, *mu, seed);
  } else {
    inelastic_.distribution->sample(E, *E_out, *mu, seed);
  }

  // Because of floating-point roundoff, it may be possible for mu to be
  // outside of the range [-1,1). In these cases, we just set mu to exactly
  // -1 or 1
  if (std::abs(*mu) > 1.0)
    *mu = std::copysign(1.0, *mu);
}

double ThermalData::get_pdf(const NuclideMicroXS& micro_xs, double E,
  double& E_out, double mu, uint64_t* seed)
{
  double pdf = -1;
  AngleEnergy* angleEnergyPtr;
  // Determine whether inelastic or elastic scattering occured
  if (prn(seed) < micro_xs.thermal_elastic / micro_xs.thermal) {
    //elastic_.distribution->get_pdf(E, *E_out, *mu, seed);
    angleEnergyPtr = elastic_.distribution.get();
  } else {
    //inelastic_.distribution->get_pdf(E, *E_out, *mu, seed);
    angleEnergyPtr = inelastic_.distribution.get();
  }
  

  if (CoherentElasticAE* coherentElasticAE = dynamic_cast<CoherentElasticAE*>(angleEnergyPtr)) {
      //  std::cout << "Used " << typeid(*coherentElasticAE).name() << " implementation." << std::endl;
        pdf = (*coherentElasticAE).get_pdf(E, E_out, mu, seed);
       // std::cout <<"pdf"<< pdf<< std::endl;
       // std::cout <<"E"<<E<< std::endl;
        //std::cout <<"mu"<< mu<< std::endl;
    //std::ofstream outFile("/root/OpenMC/projects/openmc/paper/runs/disk2/mCH2_deg60_E0.025_rho1/pdf.txt");
    //    for (double mu = -1.0; mu <= 1.0; mu += 0.001) {
    // Calculate PDF using get_pdf method
    //       double pdf1 = (*coherentElasticAE).get_pdf(E, E_out, mu, seed);    
    // Save E, E_out, mu, and pdf to the file
    //      outFile << E << "," << "," << mu << "," << pdf1 << std::endl;
    //   }
    //   outFile.close();
    bool creat_pdf_file=false;


    if (creat_pdf_file)
     {
    std::ofstream outFile("/home/itay/Documents/openmc/projects/tests/all_iso/thick_sphere/Graphite/pdf.txt");

     for (double E = 0.001; E <= 0.2; E += 0.001) {
        std::cout << "Ein" << E << std::endl;
        for (double mu = -1.0; mu <= 1.0; mu += 0.001) {
           // std::cout << "mu" << mu << std::endl;
            // Calculate PDF using get_pdf method
            //std::cout << "mu" << mu << std::endl;
            double pdf1 = (*coherentElasticAE).get_pdf(E, E_out, mu, seed);
            
            // Save E, E_out, mu, and pdf to the file
            outFile << E << "," << mu << "," << pdf1 << std::endl;
            // Update and display progress bar
                
                
        }
      

    
  }
std::cout << "Finnished" << E << std::endl;
       outFile.close();
      fatal_error("or");
     }

        // Handle CoherentElasticAE
    } else if (IncoherentElasticAE* incoherentElasticAE = dynamic_cast<IncoherentElasticAE*>(angleEnergyPtr)) {
        std::cout << "Used " << typeid(*incoherentElasticAE).name() << " implementation." << std::endl;
        // Handle IncoherentElasticAE
    } else if (IncoherentElasticAEDiscrete* incoherentElasticAEDiscrete = dynamic_cast<IncoherentElasticAEDiscrete*>(angleEnergyPtr)) {
        std::cout << "Used " << typeid(*incoherentElasticAEDiscrete).name() << " implementation." << std::endl;
        // Handle IncoherentElasticAEDiscrete
    } else if (IncoherentInelasticAEDiscrete* incoherentInelasticAEDiscrete = dynamic_cast<IncoherentInelasticAEDiscrete*>(angleEnergyPtr)) {
      //std::cout << "Used " << typeid(*incoherentInelasticAEDiscrete).name() << " implementation (water for example)" << std::endl;
        pdf = (*incoherentInelasticAEDiscrete).get_pdf(E, E_out, mu, seed,-1);
       // pdf = 0.5;
      //std::cout << "mu " << mu << std::endl;
      //std::cout << "pdf returning from secondary thermal " << pdf << std::endl;
      //std::cout << "E in " << E << std::endl;
     //std::cout << "E out " << E_out << std::endl;
     bool creat_pdf_file=false;
     if (creat_pdf_file)
     {
    std::ofstream outFile("/root/OpenMC/projects/openmc/paper/runs/thermal/runs/mwater_deg60_E0.025_rho1/pdf.txt");
    double h2o_Ein_min = 1.00000e-05;
   double h2o_Ein_max = 4.46000e+00;
   // Choose the number of steps in the logarithmic range
int num_steps = 100;  // Adjust this as needed

// Calculate the logarithmic step size
double log_step = pow(h2o_Ein_max / h2o_Ein_min, 1.0 / num_steps);
const int total_iterations = num_steps * 64 * ((2.0 / 0.01) + 1);
    int current_iteration = 0;

     for (double E = h2o_Ein_min; E <= h2o_Ein_max; E *= log_step) {
    for (int l = 0; l <= 63; ++l) {
        for (double mu = -1.0; mu <= 1.0; mu += 0.01) {
            // Calculate PDF using get_pdf method
            double pdf1 = (*incoherentInelasticAEDiscrete).get_pdf(E, E_out, mu, seed , l);
            
            // Save E, E_out, mu, and pdf to the file
            outFile << E << "," << l << "," << mu << "," << pdf1 << std::endl;
            // Update and display progress bar
                ++current_iteration;
                double progress = static_cast<double>(current_iteration) / total_iterations;
                int bar_width = 50;
                int bar_progress = static_cast<int>(progress * bar_width);

                std::cout << "\r[";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < bar_progress) {
                        std::cout << "=";
                    } else {
                        std::cout << " ";
                    }
                }

                std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << "%";
                std::cout.flush();
        }
    }
}

       outFile.close();
     }
           
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


 //std::cout << "pdf from function in thermal " << pdf << std::endl;
 return pdf;
}

void free_memory_thermal()
{
  data::thermal_scatt.clear();
  data::thermal_scatt_map.clear();
}

} // namespace openmc
