#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_38_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<38> mm_instantiator;

public:
  UnalignedM_38_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_38_MMInstantiatorWrapper__;

}  // namespace LLMMM
