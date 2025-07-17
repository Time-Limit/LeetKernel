#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_66_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<66> mm_instantiator;

public:
  UnalignedM_66_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_66_MMInstantiatorWrapper__;

}  // namespace LLMMM
