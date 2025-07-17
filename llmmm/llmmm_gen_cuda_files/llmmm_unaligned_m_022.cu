#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_22_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<22> mm_instantiator;

public:
  UnalignedM_22_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_22_MMInstantiatorWrapper__;

}  // namespace LLMMM
