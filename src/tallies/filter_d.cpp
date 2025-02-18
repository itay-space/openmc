#include "openmc/tallies/filter_d.h"

#include <algorithm>  // for min, max, copy, adjacent_find
#include <functional> // for greater_equal
#include <iterator>   // for back_inserter

#include <fmt/core.h>

#include "openmc/search.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// DFilter implementation
//==============================================================================

// Load filter parameters from an XML input file
void DFilter::from_xml(pugi::xml_node node) {
  auto bins = get_node_array<double>(node, "bins");
  this->set_bins(bins);
}

// Set bin values for the filter
void DFilter::set_bins(gsl::span<const double> bins) {
  // Clear existing bins
  bins_.clear();
  bins_.reserve(bins.size());

  // Ensure bins are sorted and unique
  if (std::adjacent_find(bins.cbegin(), bins.cend(), std::greater_equal<>()) != bins.end()) {
    throw std::runtime_error {"DFilter bins must be monotonically increasing."};
  }

  // Copy bins
  std::copy(bins.cbegin(), bins.cend(), std::back_inserter(bins_));
  n_bins_ = bins_.size() - 1;
}

// Determine which bin the particle belongs to
void DFilter::get_all_bins(const Particle& p, TallyEstimator estimator, FilterMatch& match) const {
  // Get particle properties
  double particle_time = p.time();
  double particle_velocity = p.speed(); // Assuming OpenMC provides a `speed()` function

  // Compute the modified time
  double modified_time = particle_time * particle_velocity;
  // double modified_time = 25.12;

  // If modified time is outside bin range, exit
  if (modified_time < bins_.front() || modified_time >= bins_.back()) return;

  // Find the appropriate bin
  auto i_bin = lower_bound_index(bins_.begin(), bins_.end(), modified_time);
  match.bins_.push_back(i_bin);
  match.weights_.push_back(1.0);
}

// Write filter state to a statepoint file
void DFilter::to_statepoint(hid_t filter_group) const {
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", bins_);
}

// Provide a text label for the filter (used in output files)
std::string DFilter::text_label(int bin) const {
  return fmt::format("D Filter [{}, {})", bins_[bin], bins_[bin + 1]);
}

} // namespace openmc
