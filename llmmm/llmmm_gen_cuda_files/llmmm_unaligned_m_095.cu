#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_95_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<95> mm_instantiator;

public:
  UnalignedM_95_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_95_MMInstantiatorWrapper__;

}  // namespace LLMMM
