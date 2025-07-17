#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_31_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<31> mm_instantiator;

public:
  UnalignedM_31_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_31_MMInstantiatorWrapper__;

}  // namespace LLMMM
