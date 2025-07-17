#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_109_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<109> mm_instantiator;

public:
  UnalignedM_109_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_109_MMInstantiatorWrapper__;

}  // namespace LLMMM
