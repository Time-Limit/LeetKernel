#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_74_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<74> mm_instantiator;

public:
  UnalignedM_74_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_74_MMInstantiatorWrapper__;

}  // namespace LLMMM
