#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_97_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<97> mm_instantiator;

public:
  UnalignedM_97_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_97_MMInstantiatorWrapper__;

}  // namespace LLMMM
