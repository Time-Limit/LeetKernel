#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_125_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<125> mm_instantiator;

public:
  UnalignedM_125_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_125_MMInstantiatorWrapper__;

}  // namespace LLMMM
