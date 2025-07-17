#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_83_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<83> mm_instantiator;

public:
  UnalignedM_83_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_83_MMInstantiatorWrapper__;

}  // namespace LLMMM
