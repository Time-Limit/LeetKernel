#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_6_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<6> mm_instantiator;

public:
  UnalignedM_6_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_6_MMInstantiatorWrapper__;

}  // namespace LLMMM
