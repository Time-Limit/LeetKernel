#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_99_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<99> mm_instantiator;

public:
  UnalignedM_99_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_99_MMInstantiatorWrapper__;

}  // namespace LLMMM
